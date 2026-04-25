"""
Utility functions: logging, result I/O, seeding, formatting.
"""

import os
import sys
import json
import time
import random
import logging
import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────

def setup_logging(
    name: str = "llm_inference",
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
) -> logging.Logger:
    """Set up a logger with console + optional file handler.
    Also configures the root 'llm_inference' logger so sub-modules propagate."""
    # Always configure the root llm_inference logger first
    root_logger = logging.getLogger("llm_inference")
    root_logger.setLevel(level)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if root already has handlers
    if root_logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler on root logger so all sub-modules inherit it
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root_logger.addHandler(ch)

    # File handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(
            os.path.join(log_dir, f"{name}_{ts}.log"), encoding="utf-8"
        )
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root_logger.addHandler(fh)

    return logger


# ──────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────
# Result I/O
# ──────────────────────────────────────────────────────────────────────

def save_results_json(data: Any, filepath: str) -> None:
    """Save results dictionary to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def _serialise(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_serialise)


def load_results_json(filepath: str) -> Any:
    """Load results from JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results_csv(rows: list, filepath: str, headers: Optional[list] = None) -> None:
    """Save a list of dicts (or list of lists with headers) to CSV."""
    import csv
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        if rows and isinstance(rows[0], dict):
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        else:
            writer = csv.writer(f)
            if headers:
                writer.writerow(headers)
            writer.writerows(rows)


# ──────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────

def format_memory(bytes_val: int) -> str:
    """Format bytes as human-readable string."""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 ** 2:
        return f"{bytes_val / 1024:.1f} KB"
    elif bytes_val < 1024 ** 3:
        return f"{bytes_val / (1024 ** 2):.1f} MB"
    else:
        return f"{bytes_val / (1024 ** 3):.2f} GB"


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


# ──────────────────────────────────────────────────────────────────────
# Hardware info
# ──────────────────────────────────────────────────────────────────────

def get_hardware_info() -> Dict[str, Any]:
    """Collect hardware and software environment info."""
    import platform
    import psutil

    info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        info["gpu_vram_total_gb"] = round(total_mem / (1024 ** 3), 2)
        info["gpu_compute_capability"] = (
            f"{props.major}.{props.minor}"
        )

    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        pass

    try:
        import bitsandbytes
        info["bitsandbytes_version"] = bitsandbytes.__version__
    except ImportError:
        pass

    return info


# ──────────────────────────────────────────────────────────────────────
# Timer context manager
# ──────────────────────────────────────────────────────────────────────

class Timer:
    """Simple context-manager timer."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start_time

    @property
    def elapsed_ms(self) -> float:
        return (self.elapsed or 0.0) * 1000


# ──────────────────────────────────────────────────────────────────────
# Checkpoint / resume helpers
# ──────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """Tracks completed (week, model) pairs so runs can resume after crashes.

    Stores a simple JSON file at ``results/.checkpoints/<tag>.json`` mapping
    completed keys to ``True``.  Call ``is_done(key)`` before expensive work
    and ``mark_done(key)`` after it succeeds.
    """

    def __init__(self, tag: str, results_dir: Optional[str] = None):
        if results_dir is None:
            results_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "results",
            )
        self._dir = os.path.join(results_dir, ".checkpoints")
        os.makedirs(self._dir, exist_ok=True)
        self._path = os.path.join(self._dir, f"{tag}.json")
        self._state: Dict[str, bool] = {}
        if os.path.exists(self._path):
            with open(self._path, "r", encoding="utf-8") as f:
                self._state = json.load(f)

    def is_done(self, key: str) -> bool:
        return self._state.get(key, False)

    def mark_done(self, key: str) -> None:
        self._state[key] = True
        self._flush()

    def reset(self) -> None:
        self._state = {}
        self._flush()

    def _flush(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2)

    def summary(self) -> Dict[str, bool]:
        return dict(self._state)
