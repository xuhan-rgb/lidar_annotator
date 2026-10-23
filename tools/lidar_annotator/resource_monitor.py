"""Asynchronous CPU and NVIDIA GPU utilization sampling for the Qt UI."""
from PySide6.QtCore import QObject, QProcess, QTimer, Signal


def parse_nvidia_smi_output(output):
    """Parse nvidia-smi CSV rows into (name, utilization %, used MiB, total MiB)."""
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4 or not fields[0]:
            continue
        try:
            utilization, used, total = (int(value) for value in fields[1:])
        except ValueError:
            continue
        if not (0 <= utilization <= 100 and used >= 0 and total > 0):
            continue
        rows.append((fields[0], utilization, used, total))
    return rows


def cpu_utilization(previous, current):
    """Return CPU busy percentage between /proc/stat aggregate counter samples."""
    if previous is None or len(previous) < 5 or len(current) < 5:
        return None
    deltas = [now - before for before, now in zip(previous[:8], current[:8])]
    total = sum(deltas)
    idle = deltas[3] + (deltas[4] if len(deltas) > 4 else 0)
    if total <= 0 or idle < 0 or idle > total:
        return None
    return max(0.0, min(100.0, 100.0 * (total - idle) / total))


def read_cpu_counters(path="/proc/stat"):
    try:
        with open(path, encoding="ascii") as stream:
            fields = stream.readline().split()
        if not fields or fields[0] != "cpu":
            return None
        return tuple(int(value) for value in fields[1:])
    except (OSError, ValueError):
        return None


class ResourceMonitor(QObject):
    textChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._previous_cpu = None
        self._cpu_text = "CPU --"
        self._gpu_text = "GPU不可用"
        self._process = QProcess(self)
        self._process.finished.connect(self._gpu_finished)
        self._process.errorOccurred.connect(self._gpu_error)
        self._timer = QTimer(self)
        self._timer.setInterval(2000)
        self._timer.timeout.connect(self._sample)
        self._timeout = QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.setInterval(3000)
        self._timeout.timeout.connect(self._gpu_timeout)

    def start(self):
        if not self._timer.isActive():
            self._sample()
            self._timer.start()

    def stop(self):
        self._timer.stop()
        self._timeout.stop()
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()
            self._process.waitForFinished(100)

    def _sample(self):
        current = read_cpu_counters()
        value = cpu_utilization(self._previous_cpu, current) if current is not None else None
        self._previous_cpu = current
        self._cpu_text = "CPU --" if value is None else f"CPU {value:.0f}%"
        self._emit_text()
        if self._process.state() == QProcess.ProcessState.NotRunning:
            self._process.start("nvidia-smi", ["--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"])
            self._timeout.start()

    def _gpu_finished(self, _exit_code, exit_status):
        self._timeout.stop()
        if exit_status == QProcess.ExitStatus.NormalExit:
            output = bytes(self._process.readAllStandardOutput()).decode("utf-8", "replace")
            rows = parse_nvidia_smi_output(output)
            self._gpu_text = self._format_gpus(rows) if rows else "GPU不可用"
        else:
            self._gpu_text = "GPU不可用"
        self._emit_text()

    def _gpu_error(self, _error):
        if self._process.state() == QProcess.ProcessState.NotRunning:
            self._timeout.stop()
            self._gpu_text = "GPU不可用"
            self._emit_text()

    def _gpu_timeout(self):
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()
        self._gpu_text = "GPU不可用"
        self._emit_text()

    @staticmethod
    def _format_gpus(rows):
        return " | ".join(f"GPU {name} {util}% · 显存 {used}/{total} MiB" for name, util, used, total in rows)

    def _emit_text(self):
        self.textChanged.emit(f"{self._cpu_text}   {self._gpu_text}")
