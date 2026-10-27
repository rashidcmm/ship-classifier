"""Pick the compute device, a batch size that fits, and keep an eye on GPU temperature."""
import shutil
import subprocess
import time

import torch

# batch sizes per backbone: (GPU < 6 GB, 6-11 GB, >= 12 GB)
_CUDA_BATCH = {
    "efficientnet_b0": (32, 64, 128),
    "mobilenetv3_large_100": (48, 96, 192),
    "resnet18": (48, 96, 192),
    "tiny_cnn": (64, 128, 256),
}


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def gpu_memory_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.get_device_properties(device).total_memory / 1024 ** 3


def auto_batch_size(backbone: str, device: torch.device) -> int:
    if device.type == "cpu":
        return 16
    if device.type == "mps":
        return 32
    mem = gpu_memory_gb(device) or 4.0
    small, mid, large = _CUDA_BATCH[backbone]
    return small if mem < 6 else mid if mem < 12 else large


def gpu_temperature() -> int | None:
    """NVIDIA GPU temperature in °C, or None when it cannot be read (no NVIDIA GPU, no nvidia-smi)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


class TemperatureGuard:
    """Call .step() once per batch; above max_c it pauses until the GPU cools below resume_c."""

    def __init__(self, max_c: int = 85, resume_c: int = 75, check_every: int = 50, poll_s: float = 15):
        # always cool down at least 10 °C below the limit (a limit below 75 °C would otherwise never pause)
        self.max_c, self.resume_c = max_c, min(resume_c, max_c - 10)
        self.check_every, self.poll_s = check_every, poll_s
        self.calls, self.max_seen = 0, None

    def step(self) -> float:
        self.calls += 1
        if self.calls % self.check_every:
            return 0.0
        temp = gpu_temperature()
        if temp is None:
            return 0.0
        self.max_seen = temp if self.max_seen is None else max(self.max_seen, temp)
        if temp <= self.max_c:
            return 0.0
        print(f"\nGPU at {temp} °C (limit {self.max_c} °C): pausing until it cools below {self.resume_c} °C ...")
        start = time.time()
        while temp is not None and temp >= self.resume_c:
            time.sleep(self.poll_s)
            temp = gpu_temperature()
        paused = time.time() - start
        print(f"GPU back to {temp} °C after {paused:.0f} s, continuing.")
        return paused
