import argparse
import json
import sys
import time
import re
from contextlib import nullcontext
from queue import Empty
from datetime import datetime
from pathlib import Path
from typing import TextIO

from live_plot import LiveWaveformPlot
from protocol import build_sample_id, validate_record
from capture_gate import GatedSerialReader, enable_firmware_gate, windows_key_reader


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "collected_data"
PLOT_DIRECTORY = OUTPUT_DIRECTORY / "plots"
TRIGGER_INDEX = 256
USB_SERIAL_VENDOR_IDS = {0x0403, 0x10C4, 0x1A86, 0x303A}
_reserved_output_paths: set[Path] = set()


def resolve_serial_port(requested_port: str) -> str:
    if requested_port.lower() != "auto":
        return requested_port

    try:
        from serial.tools import list_ports
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "缺少pyserial，请运行: py -m pip install -r requirements.txt"
        ) from error

    ports = list(list_ports.comports())
    candidates = [
        port for port in ports
        if port.vid in USB_SERIAL_VENDOR_IDS
    ]
    com7 = next(
        (port for port in candidates if port.device.upper() == "COM7"),
        None,
    )
    if com7 is not None:
        return com7.device
    if len(candidates) == 1:
        return candidates[0].device

    available = ", ".join(port.device for port in ports) or "无"
    raise RuntimeError(
        f"无法唯一识别ESP32串口（可见端口：{available}）；"
        "请用--port COM7指定端口"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取、验证、保存并显示木材敲击波形",
    )

    parser.add_argument(
        "--port",
        help="ESP32串口，例如COM7；可填auto自动识别；交互终端默认自动识别",
    )
    parser.add_argument("--interactive", action="store_true", help="逐项输入批量采集信息")
    parser.add_argument(
        "--hold-key", choices=("space", "f8", "none"),
        help="按住才允许保存：space空格（串口自动触发默认）、f8或none关闭",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="串口波特率，默认115200",
    )
    parser.add_argument(
        "--impact-point",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="本次采集的敲击位置，默认1",
    )
    parser.add_argument(
        "--trigger",
        choices=("auto", "software"),
        default="auto",
        help="auto等待真实敲击；software由电脑发送C触发",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="串口模式下保存多少次敲击，默认1",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="每次等待数据的超时秒数，默认60秒",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="保存PNG后弹出Matplotlib波形窗口",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="保持一个波形窗口，并在每次敲击后立即刷新",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="显示最近5秒连续预览及最近一次完整敲击",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="只保存JSONL，不生成波形PNG",
    )

    args = parser.parse_args()
    args.batch = False
    args.stop_after_count = False
    if args.interactive or (len(sys.argv) == 1 and sys.stdin.isatty()):
        configure_batch(args)

    serial_mode = bool(args.port) or sys.stdin.isatty()
    if args.hold_key is None:
        args.hold_key = "space" if serial_mode and args.trigger == "auto" else "none"
    if args.hold_key != "none" and (not serial_mode or args.trigger != "auto"):
        parser.error("--hold-key仅用于串口自动触发；管道或软件触发请使用--hold-key none")

    if args.count < 1:
        parser.error("--count必须大于等于1")

    if args.timeout <= 0:
        parser.error("--timeout必须大于0")

    if args.live and args.show:
        parser.error("--live和--show不能同时使用")

    if args.live and args.no_plot:
        parser.error("--live和--no-plot不能同时使用")

    if args.monitor and (args.show or args.live or args.no_plot):
        parser.error(
            "--monitor不能与--show、--live或--no-plot同时使用"
        )

    if args.port is None and not sys.stdin.isatty() and (
        args.trigger != "auto"
        or args.impact_point != 1
        or args.count != 1
    ):
        parser.error(
            "从标准输入读取时，--impact-point、--trigger和--count需要与--port一起使用"
        )

    return args


def configure_batch(args: argparse.Namespace) -> None:
    """实验信息由电脑端统一编号，ADC原始值和异常标记保持不变。"""
    def ask_integer(label: str, default: int, minimum: int, maximum: int | None = None) -> int:
        while True:
            value = input(f"{label} [{default}]：").strip() or str(default)
            try:
                number = int(value)
                if number >= minimum and (maximum is None or number <= maximum):
                    return number
            except ValueError:
                pass
            print("请输入允许范围内的整数。")

    print("批量采集设置：直接回车使用方括号内默认值。")
    args.port = input("COM端口 [auto]：").strip() or "auto"
    while True:
        args.stick_id = input("木头编号 [WOOD01]：").strip() or "WOOD01"
        if re.fullmatch(r"[A-Za-z0-9_-]+", args.stick_id):
            break
        print("木头编号请使用英文字母、数字、下划线或短横线。")
    args.experiment_batch = ask_integer("实验批次", 1, 1)
    args.reclamp_batch = ask_integer("重新装夹批次（未重新装夹填0）", 0, 0)
    args.impact_point = ask_integer("敲击位置（1～3）", 1, 1, 3)
    args.count = ask_integer("采集次数（异常记录也保存并计数）", 20, 1)
    args.start_strike = ask_integer("起始敲击编号（续采请接着上次编号）", 1, 1)
    args.monitor = input("开启5秒实时监控？[Y/n]：").strip().lower() != "n"
    if args.hold_key is None:
        args.hold_key = "space" if input("按住空格才允许采集？[Y/n]：").strip().lower() != "n" else "none"
    args.trigger = "auto"
    args.batch = True
    args.stop_after_count = True
    print(f"本批：{args.stick_id} / E{args.experiment_batch} / R{args.reclamp_batch} / P{args.impact_point}，共{args.count}次。")
    input("确认安装与位置正确，按Enter开始；每次敲击间隔至少1秒。")
    if args.hold_key != "none":
        print("提锤时松键；准备好后按住指定键，等待允许采集再敲击，按住直到已保存；每次松开再按只收一条。")


def create_output_path(args: argparse.Namespace | None = None) -> Path:
    """按启动时间、木头、位置和计划次数命名；独占创建防覆盖。"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    stick_id = getattr(args, "stick_id", "UNKNOWN")
    impact_point = getattr(args, "impact_point", 1)
    count = getattr(args, "count", 1)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", stick_id):
        raise ValueError("木头编号包含不能用于文件名的字符")
    # Unbounded input must not falsely claim a planned one-record batch.
    bounded = args is not None and getattr(args, "stop_after_count", False)
    bounded = bounded or (args is not None and bool(getattr(args, "port", None))
                          and not getattr(args, "monitor", False))
    count_label = f"N{count:03d}" if bounded or args is None else "Nstream"
    stem = f"{timestamp}_{stick_id}_P{impact_point:02d}_{count_label}"
    path = OUTPUT_DIRECTORY / f"{stem}.jsonl"
    sequence = 2
    while path.exists() or path.resolve() in _reserved_output_paths:
        path = OUTPUT_DIRECTORY / f"{stem}_{sequence:02d}.jsonl"
        sequence += 1
    _reserved_output_paths.add(path.resolve())
    return path


def save_plot(record: dict, show_window: bool) -> Path:
    """保存单次波形图，并按需显示窗口。"""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "缺少matplotlib，请运行: "
            "py -m pip install -r requirements.txt"
        ) from error

    sample_rate_hz = record["sample_rate_hz"]
    raw = record["raw"]
    time_ms = [
        index * 1000.0 / sample_rate_hz
        for index in range(len(raw))
    ]

    trigger_time_ms = TRIGGER_INDEX * 1000.0 / sample_rate_hz
    anomaly_text = ", ".join(record["anomaly_flags"]) or "NONE"

    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(time_ms, raw, linewidth=1.0, color="#1565C0")
    axis.axvline(
        trigger_time_ms,
        color="#D32F2F",
        linestyle="--",
        linewidth=1.2,
        label=f"Trigger ({trigger_time_ms:.2f} ms)",
    )
    axis.set_title(
        f"{record['sample_id']} | anomalies: {anomaly_text}"
    )
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("ADC raw value")
    axis.set_xlim(time_ms[0], time_ms[-1])
    axis.set_ylim(-50, 4145)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()

    PLOT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    plot_path = PLOT_DIRECTORY / f"{record['sample_id']}.png"
    figure.savefig(plot_path, dpi=160)

    if show_window:
        plt.show()

    plt.close(figure)
    return plot_path


def process_line(
    input_line: str,
    line_number: int,
    output_file: TextIO,
    sample_ids: set[str],
    create_plot: bool,
    show_window: bool,
    live_plotter: LiveWaveformPlot | None,
    batch_args: argparse.Namespace | None = None,
) -> tuple[bool, Path | None]:
    """验证并保存一行数据；返回是否有效及波形图路径。"""

    input_line = input_line.strip()

    if not input_line:
        return False, None

    try:
        record = json.loads(input_line)
        validate_record(record)

        if batch_args is not None and batch_args.batch:
            record["stick_id"] = batch_args.stick_id
            record["experiment_batch"] = batch_args.experiment_batch
            record["reclamp_batch"] = batch_args.reclamp_batch
            record["impact_point_id"] = batch_args.impact_point
            record["strike_id"] = batch_args.start_strike + len(sample_ids)
            record["sample_id"] = build_sample_id(
                record["stick_id"], record["experiment_batch"],
                record["reclamp_batch"], record["impact_point_id"], record["strike_id"],
            )
            validate_record(record)

        sample_id = record["sample_id"]

        if sample_id in sample_ids:
            raise ValueError(f"样本编号重复: {sample_id}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
            raise ValueError("样本编号包含不能用于文件名的字符")

        saved_line = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        # One impact per JSON; session JSONL remains a compatibility index.
        sample_directory = Path(output_file.name).resolve().parent / "samples"
        sample_directory.mkdir(parents=True, exist_ok=True)
        sample_path = sample_directory / f"{sample_id}.json"
        try:
            with sample_path.open("x", encoding="utf-8", newline="\n") as sample_file:
                sample_file.write(saved_line + "\n")
        except FileExistsError as error:
            raise FileExistsError(
                f"同编号数据已存在，已停止以防覆盖：{sample_path}；"
                "请调整起始敲击编号或实验批次后重新采集"
            ) from error
        sample_ids.add(sample_id)
        output_file.write(saved_line + "\n")
        output_file.flush()
        print(f"单次数据: {sample_path}", file=sys.stderr)

        plot_path = None
        try:
            if live_plotter is not None:
                plot_path = live_plotter.update(record)
            elif create_plot:
                plot_path = save_plot(record, show_window)
        except Exception as error:
            print(f"数据已保存，但绘图失败：{error}", file=sys.stderr)

        print(
            "已保存: "
            f"{sample_id}，"
            f"位置=P{record['impact_point_id']:02d}，"
            f"采样点数={record['sample_count']}，"
            f"异常={record['anomaly_flags'] or '无'}",
            file=sys.stderr,
        )

        if plot_path is not None:
            print(
                f"波形图: {plot_path.resolve()}",
                file=sys.stderr,
            )

        return True, plot_path

    except json.JSONDecodeError as error:
        print(
            f"第{line_number}行不是有效JSON: {error}",
            file=sys.stderr,
        )
    except (ValueError, RuntimeError) as error:
        print(
            f"第{line_number}行处理失败: {error}",
            file=sys.stderr,
        )

    return False, None


def send_command(device: object, command: str) -> None:
    device.write(command.encode("ascii"))
    device.flush()


def process_preview_line(
    input_line: str,
    live_plotter: LiveWaveformPlot | None,
) -> bool:
    """解析P,timestamp_us,sample,baseline,difference预览行。"""

    if not input_line.startswith("P,"):
        return False

    parts = input_line.strip().split(",")

    if len(parts) != 5:
        return True

    try:
        timestamp_us = int(parts[1])
        sample = int(parts[2])
        baseline = float(parts[3])
        difference = int(parts[4])
    except ValueError:
        return True

    if live_plotter is not None:
        live_plotter.update_preview(
            timestamp_us=timestamp_us,
            sample=sample,
            baseline=baseline,
            difference=difference,
        )

    return True


def collect_from_serial(
    args: argparse.Namespace,
    output_file: TextIO,
    sample_ids: set[str],
    live_plotter: LiveWaveformPlot | None,
) -> tuple[int, int]:
    """直接打开ESP32串口并采集指定数量的有效记录。"""

    try:
        import serial
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "缺少pyserial，请运行: "
            "py -m pip install -r requirements.txt"
        ) from error

    key_pressed = windows_key_reader(args.hold_key) if args.hold_key != "none" else None

    try:
        device = serial.Serial(
            port=args.port,
            baudrate=args.baud,
            timeout=0.01 if key_pressed is not None else 0.25,
            write_timeout=0.25,
        )
    except serial.SerialException as error:
        raise RuntimeError(
            f"无法打开串口{args.port}: {error}"
        ) from error

    with device:
        print(
            f"已打开{args.port}，等待ESP32启动……",
            file=sys.stderr,
        )

        # 打开串口通常会让ESP32自动复位
        time.sleep(2.0)
        device.reset_input_buffer()

        if key_pressed is not None:
            enable_firmware_gate(device)
            print("已确认固件支持按住许可；松键时仅预览，不启动正式采样。", file=sys.stderr)
        else:
            send_command(device, "g")

        send_command(device, str(args.impact_point))
        print(
            f"已选择敲击位置P{args.impact_point:02d}",
            file=sys.stderr,
        )

        if args.monitor or args.live:
            send_command(device, "M")
            print(
                "已启用250 Hz实时预览（上图显示最近5秒）",
                file=sys.stderr,
            )

        if args.trigger == "software":
            time.sleep(0.1)
            send_command(device, "C")
            print("已发送软件触发C", file=sys.stderr)
        else:
            if key_pressed is not None:
                print("采集已锁定。先松开按键，再提锤；按住空格或F8（按设置），等待允许采集后敲击。", file=sys.stderr)
            else:
                print("请敲击木头，正在等待自动触发……", file=sys.stderr)

        reader_context = (
            GatedSerialReader(device, key_pressed, "空格" if args.hold_key == "space" else "F8")
            if key_pressed is not None else nullcontext(None)
        )
        with reader_context as reader:
            try:
                return collect_serial_records(args, device, output_file, sample_ids, live_plotter, reader)
            finally:
                if args.monitor or args.live:
                    send_command(device, "m")


def collect_serial_records(args, device, output_file, sample_ids, live_plotter, reader):
    valid_count = invalid_count = line_number = ignored_count = 0
    deadline = time.monotonic() + args.timeout
    while (args.monitor and not args.stop_after_count) or valid_count < args.count:
        if reader is not None and live_plotter is not None:
            live_plotter.set_capture_status(reader.gate.status())
        if (
            args.monitor
            and live_plotter is not None
            and not live_plotter.is_open()
        ):
            break

        permit = None
        if reader is not None:
            try:
                item = reader.lines.get(timeout=0.05)
            except Empty:
                item = ("", None)
            if isinstance(item, Exception):
                raise RuntimeError(f"串口后台读取失败：{item}") from item
            input_line, permit = item
            raw_line = input_line.encode("utf-8")
            # 未允许采集时不消耗等待时间，留出提锤、调整装夹的时间。
            if reader.gate.ticket() is None:
                deadline = time.monotonic() + args.timeout
        else:
            raw_line = device.readline()

        if not raw_line:
            if live_plotter is not None:
                live_plotter.pump_events()

            if not args.monitor and time.monotonic() >= deadline:
                raise RuntimeError(
                    f"等待数据超过{args.timeout:g}秒"
                )
            continue

        line_number += 1
        input_line = raw_line.decode(
            "utf-8",
            errors="replace",
        )

        if process_preview_line(input_line, live_plotter):
            if not args.monitor and time.monotonic() >= deadline:
                raise RuntimeError(
                    f"等待完整敲击数据超过{args.timeout:g}秒"
                )
            continue

        if reader is not None:
            if not input_line.lstrip().startswith("{"):
                if input_line.strip():
                    print(input_line.strip(), file=sys.stderr)
                continue
            if permit is None or permit.used:
                ignored_count += 1
                continue

        valid, _ = process_line(
            input_line=input_line,
            line_number=line_number,
            output_file=output_file,
            sample_ids=sample_ids,
            create_plot=not args.no_plot,
            show_window=args.show,
            live_plotter=live_plotter,
            batch_args=args,
        )

        if not valid:
            invalid_count += 1
            continue

        valid_count += 1
        if permit is not None:
            permit.used = True
            print("本次按住已采集一条，请松键后再准备下一次。", file=sys.stderr)
        if args.batch:
            print(f"采集进度：{valid_count}/{args.count}（异常记录保留标记）", file=sys.stderr)
        deadline = time.monotonic() + args.timeout

        if (
            args.trigger == "software"
            and ((args.monitor and not args.stop_after_count) or valid_count < args.count)
        ):
            time.sleep(0.6)
            send_command(device, "C")
            print("已发送下一次软件触发C", file=sys.stderr)

    if reader is not None:
        print(f"未获按键许可而忽略的记录：{ignored_count}（不计入次数和敲击编号）", file=sys.stderr)

    return valid_count, invalid_count


def collect_from_stdin(
    args: argparse.Namespace,
    output_file: TextIO,
    sample_ids: set[str],
    live_plotter: LiveWaveformPlot | None,
) -> tuple[int, int]:
    """保留原来的模拟器/管道输入模式。"""

    valid_count = 0
    invalid_count = 0

    for line_number, input_line in enumerate(sys.stdin, start=1):
        if process_preview_line(input_line, live_plotter):
            continue

        valid, _ = process_line(
            input_line=input_line,
            line_number=line_number,
            output_file=output_file,
            sample_ids=sample_ids,
            create_plot=not args.no_plot,
            show_window=args.show,
            live_plotter=live_plotter,
        )

        if valid:
            valid_count += 1
        elif input_line.strip():
            invalid_count += 1

    return valid_count, invalid_count


def main() -> None:
    args = parse_args()

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = create_output_path(args)
    sample_ids: set[str] = set()
    live_plotter = None

    print("采集工具已启动……", file=sys.stderr)
    global PLOT_DIRECTORY
    PLOT_DIRECTORY = OUTPUT_DIRECTORY / "plots" / output_path.stem

    valid_count = 0
    invalid_count = 0

    try:
        if args.live or args.monitor:
            live_plotter = LiveWaveformPlot(PLOT_DIRECTORY)
            print(
                "实时波形窗口已打开。",
                file=sys.stderr,
            )

        with output_path.open(
            mode="x",
            encoding="utf-8",
            newline="\n",
        ) as output_file:
            if args.port or sys.stdin.isatty():
                args.port = resolve_serial_port(args.port or "auto")
                print(f"使用串口：{args.port}", file=sys.stderr)
                valid_count, invalid_count = collect_from_serial(
                    args,
                    output_file,
                    sample_ids,
                    live_plotter,
                )
            else:
                print(
                    "未指定--port，正在从标准输入读取……",
                    file=sys.stderr,
                )
                valid_count, invalid_count = collect_from_stdin(
                    args,
                    output_file,
                    sample_ids,
                    live_plotter,
                )
    except (RuntimeError, OSError, KeyboardInterrupt) as error:
        print(f"采集停止: {error}", file=sys.stderr)

    valid_count = len(sample_ids)

    if valid_count == 0:
        if output_path.exists() and output_path.stat().st_size == 0:
            output_path.unlink()
        print("没有收到有效记录，未保存数据文件。", file=sys.stderr)
        return

    print("", file=sys.stderr)
    print("采集结束", file=sys.stderr)
    print(f"有效记录: {valid_count}", file=sys.stderr)
    print(f"无效记录: {invalid_count}", file=sys.stderr)
    print(f"保存位置: {output_path.resolve()}", file=sys.stderr)

    if live_plotter is not None:
        print(
            "最后一条波形将保持显示；关闭图窗即可结束。",
            file=sys.stderr,
        )
        live_plotter.hold()


if __name__ == "__main__":
    main()
