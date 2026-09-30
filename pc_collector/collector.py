import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

from live_plot import LiveWaveformPlot
from protocol import validate_record


OUTPUT_DIRECTORY = Path("collected_data")
PLOT_DIRECTORY = OUTPUT_DIRECTORY / "plots"
TRIGGER_INDEX = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取、验证、保存并显示木材敲击波形",
    )

    parser.add_argument(
        "--port",
        help="ESP32串口，例如COM5；不指定时从标准输入读取",
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
        help="显示最近2秒连续预览及最近一次完整敲击",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="只保存JSONL，不生成波形PNG",
    )

    args = parser.parse_args()

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

    if not args.port and (
        args.trigger != "auto"
        or args.impact_point != 1
        or args.count != 1
    ):
        parser.error(
            "--impact-point、--trigger和--count需要与--port一起使用"
        )

    return args


def create_output_path() -> Path:
    """创建不会重复的数据文件名。"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return OUTPUT_DIRECTORY / f"session_{timestamp}.jsonl"


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
) -> tuple[bool, Path | None]:
    """验证并保存一行数据；返回是否有效及波形图路径。"""

    input_line = input_line.strip()

    if not input_line:
        return False, None

    try:
        record = json.loads(input_line)
        validate_record(record)

        sample_id = record["sample_id"]

        if sample_id in sample_ids:
            raise ValueError(f"样本编号重复: {sample_id}")

        sample_ids.add(sample_id)

        saved_line = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        output_file.write(saved_line + "\n")
        output_file.flush()

        plot_path = None
        if live_plotter is not None:
            plot_path = live_plotter.update(record)
        elif create_plot:
            plot_path = save_plot(record, show_window)

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

    try:
        device = serial.Serial(
            port=args.port,
            baudrate=args.baud,
            timeout=0.25,
        )
    except serial.SerialException as error:
        raise RuntimeError(
            f"无法打开串口{args.port}: {error}"
        ) from error

    valid_count = 0
    invalid_count = 0
    line_number = 0

    with device:
        print(
            f"已打开{args.port}，等待ESP32启动……",
            file=sys.stderr,
        )

        # 打开串口通常会让ESP32自动复位
        time.sleep(2.0)
        device.reset_input_buffer()

        send_command(device, str(args.impact_point))
        print(
            f"已选择敲击位置P{args.impact_point:02d}",
            file=sys.stderr,
        )

        if args.monitor:
            send_command(device, "M")
            print(
                "已启用250 Hz实时预览（上图显示最近2秒）",
                file=sys.stderr,
            )

        if args.trigger == "software":
            time.sleep(0.1)
            send_command(device, "C")
            print("已发送软件触发C", file=sys.stderr)
        else:
            print("请敲击木头，正在等待自动触发……", file=sys.stderr)

        deadline = time.monotonic() + args.timeout

        while valid_count < args.count:
            raw_line = device.readline()

            if not raw_line:
                if live_plotter is not None:
                    live_plotter.pump_events()

                if time.monotonic() >= deadline:
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
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"等待完整敲击数据超过{args.timeout:g}秒"
                    )
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

            if not valid:
                invalid_count += 1
                continue

            valid_count += 1
            deadline = time.monotonic() + args.timeout

            if (
                args.trigger == "software"
                and valid_count < args.count
            ):
                time.sleep(0.6)
                send_command(device, "C")
                print("已发送下一次软件触发C", file=sys.stderr)

        if args.monitor:
            send_command(device, "m")

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
    output_path = create_output_path()
    sample_ids: set[str] = set()
    live_plotter = None

    print("采集工具已启动……", file=sys.stderr)

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
            mode="w",
            encoding="utf-8",
            newline="\n",
        ) as output_file:
            if args.port:
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
    except (RuntimeError, KeyboardInterrupt) as error:
        print(f"采集停止: {error}", file=sys.stderr)

    if valid_count == 0:
        output_path.unlink(missing_ok=True)
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
