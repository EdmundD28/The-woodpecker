"""Windows 按住采集控制；后台持续读串口，避免绘图期间积压旧记录。"""

import ctypes
import os
import sys
import threading
import time
from dataclasses import dataclass
from queue import Queue


ARM_DELAY_SECONDS = 1.0
MAX_LINE_BYTES = 65536
HEARTBEAT_SECONDS = 0.05


def enable_firmware_gate(device, timeout: float = 3.0) -> None:
    """必须确认固件支持许可，不能把电脑丢弃数据冒充为禁止触发。"""
    device.write(b"G")
    device.flush()
    deadline = time.monotonic() + timeout
    pending = bytearray()
    while time.monotonic() < deadline:
        pending.extend(device.read(min(max(device.in_waiting, 1), 4096)))
        while b"\n" in pending:
            line, _, remainder = pending.partition(b"\n")
            pending = bytearray(remainder)
            if line.strip() == b"#GATE,V1":
                return
        if len(pending) > MAX_LINE_BYTES:
            pending.clear()
    raise RuntimeError(
        "固件未确认按住采集功能。请先上传本次更新的firmware_a_sample；"
        "旧固件只支持自动触发，不能保证松键时只预览。"
    )


def windows_key_reader(key: str):
    if os.name != "nt":
        raise RuntimeError("按住采集目前仅支持Windows；其他系统请使用--hold-key none")
    virtual_key = {"space": 0x20, "f8": 0x77}[key]
    get_key_state = ctypes.WinDLL("user32", use_last_error=True).GetAsyncKeyState
    get_key_state.argtypes = [ctypes.c_int]
    get_key_state.restype = ctypes.c_short
    return lambda: bool(get_key_state(virtual_key) & 0x8000)


@dataclass
class HoldSession:
    used: bool = False


class HoldGate:
    """一次按住最多保存一条；接收中松开或重新按下均使当前行作废。"""

    def __init__(self, key_label: str) -> None:
        self.key_label = key_label
        self.session = None
        self.ready_at = 0.0
        self.ready = False
        self.released_once = False

    def update(self, pressed: bool, now: float) -> None:
        if not pressed:
            self.released_once = True
            if self.session is not None:
                print(f"采集已锁定；提锤后按住{self.key_label}。", file=sys.stderr)
            self.session = None
            self.ready = False
        elif self.released_once:
            if self.session is None:
                self.session = HoldSession()
                self.ready_at = now + ARM_DELAY_SECONDS
                print("正在排除旧数据，请继续按住，等待允许采集……", file=sys.stderr)
            if not self.ready and now >= self.ready_at:
                self.ready = True
                print(f"允许采集：保持按住{self.key_label}并敲击，直到提示已保存。", file=sys.stderr)

    def ticket(self):
        session = self.session
        if self.ready and session is not None and not session.used:
            return session
        return None

    def discard_old_record(self, now: float) -> None:
        # 完整旧记录到达后再留出一秒，不能在旧JSON传输中途开放。
        if self.session is not None and not self.session.used:
            self.ready = False
            self.ready_at = now + ARM_DELAY_SECONDS

    def status(self) -> str:
        session = self.session
        if session is None:
            return f"已锁定：提锤后按住{self.key_label}"
        if session.used:
            return "本次已采集：请松键后再准备下一次"
        if not self.ready:
            return "正在排除旧数据：请保持按住，暂勿敲击"
        return f"允许采集：按住{self.key_label}，敲击后等待已保存"


class GatedSerialReader:
    """约每10 ms检查按键，整行接收期间始终获得许可才交给保存逻辑。"""

    def __init__(self, device, key_pressed, key_label: str) -> None:
        self.device = device
        self.key_pressed = key_pressed
        self.gate = HoldGate(key_label)
        self.lines = Queue()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.buffer = bytearray()
        self.line_ticket = None
        self.overflow = False
        self.last_command = None
        self.last_heartbeat = 0.0

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stopped.set()
        self.thread.join(timeout=1.0)
        self.device.write(b"a")
        self.device.flush()

    def sync_permission(self, now: float) -> None:
        command = b"A" if self.gate.ticket() is not None else b"a"
        if command != self.last_command or (command == b"A" and now - self.last_heartbeat >= HEARTBEAT_SECONDS):
            self.device.write(command)
            self.device.flush()
            self.last_command = command
            self.last_heartbeat = now

    def feed(self, data: bytes, now: float) -> None:
        """每行在首字节处绑定按住会话；跨越松键的行不能重新获准。"""
        ticket = self.gate.ticket()
        if self.buffer and self.line_ticket is not ticket:
            self.line_ticket = None
        for value in data:
            if not self.buffer and not self.overflow:
                self.line_ticket = self.gate.ticket()
            if value == 10:
                if not self.overflow:
                    line = self.buffer.decode("utf-8", errors="replace")
                    permit = self.line_ticket
                    if line.lstrip().startswith("{") and permit is None:
                        self.gate.discard_old_record(now)
                    self.lines.put((line, permit))
                self.buffer.clear()
                self.line_ticket = None
                self.overflow = False
            elif not self.overflow:
                self.buffer.append(value)
                if len(self.buffer) > MAX_LINE_BYTES:
                    self.buffer.clear()
                    self.line_ticket = None
                    self.overflow = True

    def _run(self) -> None:
        try:
            while not self.stopped.is_set():
                self.gate.update(self.key_pressed(), time.monotonic())
                self.sync_permission(time.monotonic())
                # 即使未按键，也要持续排空串口。按下后不能回收此前积压的波形。
                data = self.device.read(min(max(self.device.in_waiting, 1), 4096))
                self.gate.update(self.key_pressed(), time.monotonic())
                self.feed(data, time.monotonic())
                self.sync_permission(time.monotonic())
        except Exception as error:
            self.lines.put(error)
