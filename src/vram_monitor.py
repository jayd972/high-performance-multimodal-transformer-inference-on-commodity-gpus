"""
GPU VRAM and CPU RAM monitoring utilities.

Uses pynvml for GPU metrics and psutil for CPU/system RAM.
Provides a polling monitor for continuous tracking and
point-in-time snapshot functions.
"""

import time
import threading
from typing import Dict, Optional, List
from dataclasses import dataclass, field

import torch
import psutil

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


@dataclass
class MemorySnapshot:
    """A single memory measurement."""
    timestamp: float
    gpu_used_mb: float = 0.0
    gpu_total_mb: float = 0.0
    gpu_free_mb: float = 0.0
    cpu_ram_used_mb: float = 0.0
    cpu_ram_total_mb: float = 0.0
    torch_allocated_mb: float = 0.0
    torch_reserved_mb: float = 0.0


class VRAMMonitor:
    """
    Monitors GPU VRAM and CPU RAM usage.
    
    Can run in background polling mode or take point-in-time snapshots.
    """

    def __init__(self, gpu_index: int = 0, poll_interval_s: float = 0.1):
        self.gpu_index = gpu_index
        self.poll_interval_s = poll_interval_s
        self._snapshots: List[MemorySnapshot] = []
        self._polling = False
        self._poll_thread: Optional[threading.Thread] = None
        self._nvml_handle = None

        # Initialise pynvml
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                self._nvml_handle = None

    def snapshot(self) -> MemorySnapshot:
        """Take a single memory snapshot."""
        snap = MemorySnapshot(timestamp=time.time())

        # GPU via pynvml
        if self._nvml_handle is not None:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                snap.gpu_used_mb = mem_info.used / (1024 ** 2)
                snap.gpu_total_mb = mem_info.total / (1024 ** 2)
                snap.gpu_free_mb = mem_info.free / (1024 ** 2)
            except Exception:
                pass

        # GPU via PyTorch
        if torch.cuda.is_available():
            snap.torch_allocated_mb = torch.cuda.memory_allocated(self.gpu_index) / (1024 ** 2)
            snap.torch_reserved_mb = torch.cuda.memory_reserved(self.gpu_index) / (1024 ** 2)

        # CPU RAM
        vm = psutil.virtual_memory()
        snap.cpu_ram_used_mb = vm.used / (1024 ** 2)
        snap.cpu_ram_total_mb = vm.total / (1024 ** 2)

        return snap

    def start_polling(self) -> None:
        """Start background memory polling."""
        if self._polling:
            return
        self._polling = True
        self._snapshots.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def stop_polling(self) -> List[MemorySnapshot]:
        """Stop polling and return collected snapshots."""
        self._polling = False
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
            self._poll_thread = None
        return list(self._snapshots)

    def _poll_loop(self) -> None:
        while self._polling:
            self._snapshots.append(self.snapshot())
            time.sleep(self.poll_interval_s)

    def get_peak_gpu_mb(self) -> float:
        """Peak GPU memory from collected snapshots."""
        if not self._snapshots:
            return 0.0
        return max(s.gpu_used_mb for s in self._snapshots)

    def get_peak_torch_allocated_mb(self) -> float:
        """Peak PyTorch allocated memory from collected snapshots."""
        if not self._snapshots:
            return 0.0
        return max(s.torch_allocated_mb for s in self._snapshots)

    def get_peak_cpu_ram_mb(self) -> float:
        """Peak CPU RAM from collected snapshots."""
        if not self._snapshots:
            return 0.0
        return max(s.cpu_ram_used_mb for s in self._snapshots)

    def reset(self) -> None:
        """Clear all snapshots."""
        self._snapshots.clear()

    def get_summary(self) -> Dict:
        """Summarise memory usage from collected snapshots."""
        if not self._snapshots:
            return {"error": "No snapshots collected"}
        return {
            "num_snapshots": len(self._snapshots),
            "peak_gpu_used_mb": round(self.get_peak_gpu_mb(), 1),
            "peak_torch_allocated_mb": round(self.get_peak_torch_allocated_mb(), 1),
            "peak_cpu_ram_used_mb": round(self.get_peak_cpu_ram_mb(), 1),
            "gpu_total_mb": round(self._snapshots[0].gpu_total_mb, 1),
            "duration_s": round(
                self._snapshots[-1].timestamp - self._snapshots[0].timestamp, 2
            ),
        }

    def __del__(self):
        self._polling = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────
# Convenience functions
# ──────────────────────────────────────────────────────────────────────

def get_current_gpu_memory() -> Dict[str, float]:
    """Quick GPU memory check (MB)."""
    result = {}
    if torch.cuda.is_available():
        result["torch_allocated_mb"] = round(
            torch.cuda.memory_allocated() / (1024 ** 2), 1
        )
        result["torch_reserved_mb"] = round(
            torch.cuda.memory_reserved() / (1024 ** 2), 1
        )
    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            result["gpu_used_mb"] = round(mem.used / (1024 ** 2), 1)
            result["gpu_free_mb"] = round(mem.free / (1024 ** 2), 1)
            result["gpu_total_mb"] = round(mem.total / (1024 ** 2), 1)
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return result


def reset_peak_memory() -> None:
    """Reset PyTorch CUDA peak memory stats."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def get_peak_memory_mb() -> float:
    """Get PyTorch peak allocated memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0
