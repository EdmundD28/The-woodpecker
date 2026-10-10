from collections import deque
from pathlib import Path
from time import monotonic


TRIGGER_INDEX = 256
PREVIEW_WINDOW_SECONDS = 5.0
PREVIEW_REDRAW_INTERVAL_SECONDS = 0.05


class LiveWaveformPlot:
    """显示连续预览和最近一次完整敲击波形。"""

    def __init__(self, plot_directory: Path) -> None:
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "缺少matplotlib，请运行: "
                "py -m pip install -r requirements.txt"
            ) from error

        self.plt = plt
        self.plot_directory = plot_directory
        self.preview_times: deque[float] = deque()
        self.preview_samples: deque[int] = deque()
        self.preview_baselines: deque[float] = deque()
        self.last_preview_redraw = 0.0
        self.capture_status = None

        self.plt.ion()
        self.figure, axes = self.plt.subplots(
            2,
            1,
            figsize=(11, 8),
        )
        self.preview_axis = axes[0]
        self.capture_axis = axes[1]

        (self.preview_line,) = self.preview_axis.plot(
            [],
            [],
            linewidth=1.0,
            color="#1565C0",
            label="ADC preview",
        )
        (self.baseline_line,) = self.preview_axis.plot(
            [],
            [],
            linewidth=1.0,
            color="#616161",
            linestyle="--",
            label="Baseline",
        )
        self.preview_axis.set_title(
            "Live preview: waiting for ESP32 monitor data..."
        )
        self.preview_axis.set_xlabel("Time relative to now (s)")
        self.preview_axis.set_ylabel("ADC raw value")
        self.preview_axis.set_xlim(-PREVIEW_WINDOW_SECONDS, 0.0)
        self.preview_axis.set_ylim(-50, 4145)
        self.preview_axis.grid(True, alpha=0.25)
        self.preview_axis.legend(loc="upper right")

        (self.capture_line,) = self.capture_axis.plot(
            [],
            [],
            linewidth=1.0,
            color="#2E7D32",
            label="Captured waveform",
        )
        self.trigger_line = self.capture_axis.axvline(
            16.0,
            color="#D32F2F",
            linestyle="--",
            linewidth=1.2,
            label="Trigger",
        )
        self.capture_axis.set_title(
            "Latest 16 kHz capture: waiting for an impact..."
        )
        self.capture_axis.set_xlabel("Time (ms)")
        self.capture_axis.set_ylabel("ADC raw value")
        self.capture_axis.set_xlim(0.0, 64.0)
        self.capture_axis.set_ylim(-50, 4145)
        self.capture_axis.grid(True, alpha=0.25)
        self.capture_axis.legend(loc="upper right")

        self.figure.tight_layout()

        try:
            self.figure.canvas.manager.set_window_title(
                "Wood Impact Monitor"
            )
        except AttributeError:
            pass

        self.figure.show()
        self.pump_events()

    def is_open(self) -> bool:
        return self.plt.fignum_exists(self.figure.number)

    def set_capture_status(self, status: str) -> None:
        """在图窗标题栏显示按住采集状态，避免必须切回终端查看。"""
        if status == self.capture_status or not self.is_open():
            return
        self.capture_status = status
        try:
            self.figure.canvas.manager.set_window_title(f"木材采样 | {status}")
        except AttributeError:
            pass

    def pump_events(self) -> None:
        """在等待串口数据时保持窗口可响应。"""

        if not self.is_open():
            return

        self.figure.canvas.flush_events()
        self.plt.pause(0.001)

    def update_preview(
        self,
        timestamp_us: int,
        sample: int,
        baseline: float,
        difference: int,
    ) -> None:
        """在上半图中更新最近两秒的低速预览。"""

        if not self.is_open():
            return

        timestamp_seconds = timestamp_us / 1_000_000.0
        self.preview_times.append(timestamp_seconds)
        self.preview_samples.append(sample)
        self.preview_baselines.append(baseline)

        oldest_allowed = (
            timestamp_seconds - PREVIEW_WINDOW_SECONDS
        )

        while (
            self.preview_times
            and self.preview_times[0] < oldest_allowed
        ):
            self.preview_times.popleft()
            self.preview_samples.popleft()
            self.preview_baselines.popleft()

        if monotonic() - self.last_preview_redraw < PREVIEW_REDRAW_INTERVAL_SECONDS:
            return

        relative_times = [
            value - timestamp_seconds
            for value in self.preview_times
        ]

        self.preview_line.set_data(
            relative_times,
            list(self.preview_samples),
        )
        self.baseline_line.set_data(
            relative_times,
            list(self.preview_baselines),
        )
        self.preview_axis.set_title(
            "Live preview (5 s) | "
            f"ADC={sample} | baseline={baseline:.1f} | "
            f"difference={difference}"
        )
        self.figure.canvas.draw_idle()
        self.pump_events()
        self.last_preview_redraw = monotonic()

    def update(self, record: dict) -> Path | None:
        """在下半图显示最新完整敲击，并保存组合图。"""

        if not self.is_open():
            return None

        sample_rate_hz = record["sample_rate_hz"]
        raw = record["raw"]
        time_ms = [
            index * 1000.0 / sample_rate_hz
            for index in range(len(raw))
        ]
        trigger_time_ms = (
            TRIGGER_INDEX * 1000.0 / sample_rate_hz
        )
        anomaly_text = (
            ", ".join(record["anomaly_flags"]) or "NONE"
        )

        self.capture_line.set_data(time_ms, raw)
        self.trigger_line.set_xdata(
            [trigger_time_ms, trigger_time_ms]
        )
        self.trigger_line.set_label(
            f"Trigger ({trigger_time_ms:.2f} ms)"
        )
        self.capture_axis.set_title(
            f"{record['sample_id']} | anomalies: {anomaly_text}"
        )
        self.capture_axis.set_xlim(time_ms[0], time_ms[-1])
        self.capture_axis.legend(loc="upper right")
        self.figure.canvas.draw_idle()
        self.pump_events()

        self.plot_directory.mkdir(parents=True, exist_ok=True)
        plot_path = (
            self.plot_directory / f"{record['sample_id']}.png"
        )
        self.figure.savefig(plot_path, dpi=160)
        return plot_path

    def hold(self) -> None:
        """采集结束后保持最后画面，直到用户关闭窗口。"""

        if not self.is_open():
            return

        self.plt.ioff()
        self.plt.show()
