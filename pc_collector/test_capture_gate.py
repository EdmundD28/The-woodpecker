import argparse
import copy
import io
import json
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from queue import Empty, Queue
from unittest.mock import Mock, patch

from capture_gate import GatedSerialReader, HoldGate, HoldSession, MAX_LINE_BYTES, enable_firmware_gate
from collector import collect_serial_records, parse_args


class GateTests(unittest.TestCase):
    def setUp(self):
        self.log = io.StringIO()
        self.redirect = redirect_stderr(self.log)
        self.redirect.__enter__()
        self.reader = GatedSerialReader(None, None, "空格")
        self.gate = self.reader.gate
        self.gate.update(False, 0)

    def tearDown(self):
        self.redirect.__exit__(None, None, None)

    def arm(self, start=0):
        self.gate.update(True, start)
        self.gate.update(True, start + 1.01)
        return self.gate.ticket()

    def test_locked_and_warming_records_are_rejected(self):
        self.reader.feed(b'{"raw":[]}\n', 0)
        self.gate.update(True, 1)
        self.reader.feed(b'{"raw":[]}\n', 1.2)
        self.assertIsNone(self.reader.lines.get()[1])
        self.assertIsNone(self.reader.lines.get()[1])
        self.gate.update(True, 2.1)
        self.assertIsNone(self.gate.ticket())
        self.gate.update(True, 2.21)
        self.assertIsNotNone(self.gate.ticket())

    def test_fragmented_record_is_assembled(self):
        ticket = self.arm()
        self.reader.feed(b'{"raw":', 1.1)
        self.reader.feed(b'[]}\r\n', 1.2)
        line, permit = self.reader.lines.get()
        self.assertEqual(json.loads(line), {"raw": []})
        self.assertIs(permit, ticket)

    def test_release_mid_record_rejects_it(self):
        self.arm()
        self.reader.feed(b'{"raw":', 1.1)
        self.gate.update(False, 1.2)
        self.reader.feed(b'[]}\n', 1.3)
        self.assertIsNone(self.reader.lines.get()[1])

    def test_repress_cannot_rescue_an_old_partial_record(self):
        self.arm()
        self.reader.feed(b'{"raw":', 1.1)
        self.gate.update(False, 1.2)
        self.arm(2)
        self.reader.feed(b'[]}\n', 3.1)
        self.assertIsNone(self.reader.lines.get()[1])
        self.assertIsNone(self.gate.ticket())

    def test_record_started_during_warmup_is_never_accepted(self):
        self.gate.update(True, 0)
        self.reader.feed(b'{"raw":', 0.5)
        self.gate.update(True, 1.1)
        self.reader.feed(b'[]}\n{"raw":[]}\n', 1.2)
        self.assertIsNone(self.reader.lines.get()[1])
        self.assertIsNone(self.reader.lines.get()[1])
        self.assertIsNone(self.gate.ticket())

    def test_starting_with_key_down_requires_release(self):
        gate = HoldGate("空格")
        gate.update(True, 0)
        gate.update(True, 10)
        self.assertIsNone(gate.ticket())
        gate.update(False, 11)
        gate.update(True, 12)
        gate.update(True, 13.01)
        self.assertIsNotNone(gate.ticket())

    def test_completed_record_survives_release_before_plotting(self):
        ticket = self.arm()
        self.reader.feed(b'{"raw":[]}\n', 1.1)
        self.gate.update(False, 1.2)
        self.assertIs(self.reader.lines.get()[1], ticket)

    def test_used_session_requires_release_to_rearm(self):
        ticket = self.arm()
        ticket.used = True
        self.gate.update(True, 5)
        self.assertIsNone(self.gate.ticket())
        self.gate.update(False, 6)
        self.assertIsNot(self.arm(7), ticket)

    def test_preview_is_available_while_locked(self):
        self.reader.feed(b'P,100,2048,2048.0,0\n', 0)
        self.assertTrue(self.reader.lines.get()[0].startswith("P,"))

    def test_overlong_line_is_discarded_and_next_line_recovers(self):
        self.arm()
        self.reader.feed(b'x' * (MAX_LINE_BYTES + 1), 1.1)
        self.reader.feed(b'\n{"raw":[]}\n', 1.2)
        self.assertEqual(self.reader.lines.qsize(), 1)
        self.assertIsNotNone(self.reader.lines.get()[1])

    def test_serial_failure_is_forwarded(self):
        device = Mock()
        device.read.side_effect = OSError("disconnected")
        device.in_waiting = 0
        reader = GatedSerialReader(device, lambda: False, "空格")
        with reader:
            self.assertIsInstance(reader.lines.get(timeout=1), OSError)

    def test_permission_heartbeat_and_release(self):
        device = Mock()
        self.reader.device = device
        self.reader.sync_permission(0)
        self.arm()
        self.reader.sync_permission(1.1)
        self.reader.sync_permission(1.11)
        self.reader.sync_permission(1.16)
        self.gate.update(False, 1.17)
        self.reader.sync_permission(1.17)
        self.assertEqual([call.args[0] for call in device.write.call_args_list], [b'a', b'A', b'A', b'a'])

    def test_firmware_handshake_accepts_fragmented_confirmation(self):
        device = Mock()
        device.in_waiting = 1
        device.read.side_effect = [b'old data\n#GA', b'TE,V1\r', b'\n']
        enable_firmware_gate(device)
        device.write.assert_called_once_with(b'G')

    def test_old_firmware_is_rejected(self):
        device = Mock()
        with self.assertRaisesRegex(RuntimeError, "请先上传"):
            enable_firmware_gate(device, timeout=0)

    def test_background_reader_keeps_preview_and_rejects_release_during_frame(self):
        class Device:
            def __init__(self):
                self.input = Queue()
                self.commands = []
                self.in_waiting = 0

            def read(self, size):
                try:
                    return self.input.get(timeout=0.005)
                except Empty:
                    return b""

            def write(self, data):
                self.commands.append(data)

            def flush(self):
                pass

        def wait_until(condition):
            deadline = time.monotonic() + 1
            while not condition():
                if time.monotonic() > deadline:
                    self.fail("后台串口读取未按预期推进")
                time.sleep(0.005)

        device = Device()
        pressed = threading.Event()
        reader = GatedSerialReader(device, pressed.is_set, "空格")
        with patch("capture_gate.ARM_DELAY_SECONDS", 0.02), reader:
            device.input.put(b'P,1,2048,2048.0,0\n')
            self.assertTrue(reader.lines.get(timeout=1)[0].startswith("P,"))
            pressed.set()
            wait_until(lambda: reader.gate.ticket() is not None)
            device.input.put(b'{"raw":')
            wait_until(lambda: bool(reader.buffer))
            pressed.clear()
            wait_until(lambda: reader.gate.ticket() is None)
            device.input.put(b'[]}\nP,2,2048,2048.0,0\n')
            self.assertIsNone(reader.lines.get(timeout=1)[1])
            self.assertTrue(reader.lines.get(timeout=1)[0].startswith("P,"))
            pressed.set()
            wait_until(lambda: reader.gate.ticket() is not None)
            device.input.put(b'{"raw":[]}\n')
            self.assertIsNotNone(reader.lines.get(timeout=1)[1])
        self.assertEqual(device.commands[-1], b'a')
        self.assertFalse(reader.thread.is_alive())


class CollectorTests(unittest.TestCase):
    def test_only_permitted_records_are_saved_and_numbered(self):
        source = Path(__file__).parent / "examples" / "WOOD_IMPACT_V1_example.jsonl"
        record = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
        record["anomaly_flags"] = ["WEAK_SIGNAL"]
        first, second = HoldSession(), HoldSession()
        reader = Mock()
        reader.gate = HoldGate("空格")
        reader.lines = Queue()
        for permit, line in [
            (None, "P,1,2048,2048.0,0"),
            (None, json.dumps(record)),
            (first, "{broken"),
            (first, json.dumps(record)),
            (first, json.dumps(record)),
            (None, json.dumps(record)),
            (second, json.dumps(record)),
        ]:
            reader.lines.put((line, permit))
        args = argparse.Namespace(
            timeout=60, monitor=False, stop_after_count=True, count=2,
            no_plot=True, show=False, trigger="auto", batch=True,
            stick_id="TEST", experiment_batch=2, reclamp_batch=0,
            impact_point=2, start_strike=11,
        )
        original = copy.deepcopy(record)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory, redirect_stderr(io.StringIO()) as log:
            path = Path(directory) / "session.jsonl"
            with path.open("w", encoding="utf-8") as output:
                counts = collect_serial_records(args, Mock(), output, set(), None, reader)
            saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(counts, (2, 1))
            self.assertEqual([r["strike_id"] for r in saved], [11, 12])
            self.assertEqual([r["sample_id"] for r in saved], ["TEST-E02-R00-P02-H011", "TEST-E02-R00-P02-H012"])
            self.assertEqual(len(list((Path(directory) / "samples").glob("*.json"))), 2)
            for result in saved:
                self.assertEqual(result["raw"], original["raw"])
                self.assertEqual(result["anomaly_flags"], ["WEAK_SIGNAL"])
            self.assertIn("忽略的记录：3", log.getvalue())
            self.assertTrue(first.used and second.used)

    def test_cli_defaults_and_overrides(self):
        with patch("sys.stdin.isatty", return_value=False):
            for options, expected in [
                (["--port", "COM7"], "space"),
                (["--port", "COM7", "--hold-key", "f8"], "f8"),
                (["--port", "COM7", "--hold-key", "none"], "none"),
                (["--port", "COM7", "--trigger", "software"], "none"),
                (["--no-plot"], "none"),
            ]:
                with self.subTest(options=options), patch("sys.argv", ["collector.py"] + options):
                    self.assertEqual(parse_args().hold_key, expected)
            with patch("sys.argv", ["collector.py", "--hold-key", "space"]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()


if __name__ == "__main__":
    unittest.main()
