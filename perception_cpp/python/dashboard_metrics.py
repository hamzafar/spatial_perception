import time
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


class DashboardMetrics:

    def __init__(self):
        self.last_frame_time = None
        self.fps = 0.0

        # GPU handle
        self.gpu_handle = None

        if NVML_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.gpu_handle = None

    # -------------------------------------------------

    def update_fps(self):
        now = time.perf_counter()

        if self.last_frame_time is not None:
            dt = now - self.last_frame_time

            if dt > 0:
                instant_fps = 1.0 / dt

                # Smooth FPS
                self.fps = (
                    0.9 * self.fps +
                    0.1 * instant_fps
                )

        self.last_frame_time = now

        return self.fps

    # -------------------------------------------------

    def get_cpu(self):
        return psutil.cpu_percent(interval=None)

    # -------------------------------------------------

    def get_gpu(self):
        if self.gpu_handle is None:
            return 0.0

        try:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(
                self.gpu_handle
            )

            return float(utilization.gpu)

        except Exception:
            return 0.0

    # -------------------------------------------------

    def get_metrics(self, latency_ms=None):

        fps = self.update_fps()

        return {
            "fps": round(fps, 1),
            "latency_ms": round(latency_ms, 1)
                if latency_ms is not None else 0.0,
            "gpu_pct": round(self.get_gpu(), 1),
            "cpu_pct": round(self.get_cpu(), 1),
        }