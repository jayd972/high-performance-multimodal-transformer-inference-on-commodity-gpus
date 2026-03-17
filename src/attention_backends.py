"""
Attention backend selection and benchmarking.

Manages PyTorch SDPA backend selection (flash, memory-efficient, math)
and logs which backend actually ran.

Supports both the new ``torch.nn.attention.sdpa_kernel`` API (PyTorch ≥ 2.5)
and the legacy ``torch.backends.cuda.sdp_kernel`` API for older versions.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional
from contextlib import contextmanager

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("llm_inference.attention")

# ──────────────────────────────────────────────────────────────────────
# API detection — prefer torch.nn.attention.sdpa_kernel (PyTorch ≥ 2.5)
# ──────────────────────────────────────────────────────────────────────

_USE_NEW_SDPA_API: bool = (
    hasattr(torch, "nn")
    and hasattr(torch.nn, "attention")
    and hasattr(torch.nn.attention, "sdpa_kernel")
)

if _USE_NEW_SDPA_API:
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel_fn, SDPBackend
    _BACKEND_ENUM = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }
else:
    _sdpa_kernel_fn = None
    _BACKEND_ENUM = {}


def _sdpa_kernel_ctx(backend: str):
    """Return the appropriate SDPA kernel context manager for *backend*."""
    if _USE_NEW_SDPA_API:
        return _sdpa_kernel_fn(_BACKEND_ENUM[backend])

    # Legacy API (PyTorch < 2.5)
    return torch.backends.cuda.sdp_kernel(
        enable_flash=(backend == "flash"),
        enable_mem_efficient=(backend == "mem_efficient"),
        enable_math=(backend == "math"),
    )


# ──────────────────────────────────────────────────────────────────────
# Backend availability detection
# ──────────────────────────────────────────────────────────────────────

def check_attention_backends() -> Dict[str, Any]:
    """
    Check which SDPA backends are available on this hardware.

    Returns a dict mapping backend name to availability info.
    """
    info: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "sdpa_api": (
            "torch.nn.attention.sdpa_kernel"
            if _USE_NEW_SDPA_API
            else "torch.backends.cuda.sdp_kernel (legacy)"
        ),
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["compute_capability"] = f"{props.major}.{props.minor}"
        info["sdpa_available"] = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )

        info["flash_sdp_available"] = _test_sdpa_backend("flash")
        info["mem_efficient_sdp_available"] = _test_sdpa_backend("mem_efficient")
        info["math_sdp_available"] = True  # always available as fallback

        try:
            import flash_attn  # type: ignore
            info["flash_attn_version"] = flash_attn.__version__
            info["flash_attn_2_available"] = True
        except ImportError:
            info["flash_attn_2_available"] = False

    return info


def _test_sdpa_backend(backend: str) -> bool:
    """Test if a specific SDPA backend works with shapes close to real models.

    Uses (batch=1, heads=8, seq_len=32, head_dim=128) with ``is_causal=True``
    to exercise the same code-path that a transformer decoder layer would hit.
    """
    if not torch.cuda.is_available():
        return False
    try:
        q = torch.randn(1, 8, 32, 128, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 8, 32, 128, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 8, 32, 128, device="cuda", dtype=torch.float16)

        with _sdpa_kernel_ctx(backend):
            torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True,
            )
        return True
    except Exception as e:
        logger.debug(f"Backend '{backend}' test failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────
# Backend context managers
# ──────────────────────────────────────────────────────────────────────

@contextmanager
def sdpa_backend(backend: str = "default"):
    """
    Context manager to select a specific SDPA backend.

    Args:
        backend: One of "flash", "mem_efficient", "math", "default", "eager".

    Exceptions raised by code running inside the ``with`` block propagate
    normally — callers should catch them if per-backend failure is expected.
    """
    if backend in ("default", "eager"):
        yield
        return

    with _sdpa_kernel_ctx(backend):
        yield


def load_model_with_attention(
    model_id: str,
    attention_backend: str = "sdpa",
) -> tuple:
    """
    Load a model with a specific attention implementation.

    Args:
        model_id: HuggingFace model ID.
        attention_backend: "sdpa", "flash_attention_2", "eager".
    """
    from src.model_loader import load_model_4bit

    attn_map = {
        "sdpa": "sdpa",
        "sdpa_flash": "sdpa",
        "sdpa_mem_efficient": "sdpa",
        "sdpa_math": "sdpa",
        "flash_attention_2": "flash_attention_2",
        "eager": "eager",
    }
    attn_impl = attn_map.get(attention_backend, "sdpa")

    logger.info(f"Loading model with attn_implementation='{attn_impl}'")
    model, tokenizer = load_model_4bit(
        model_id,
        attn_implementation=attn_impl,
    )

    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────
# Attention benchmark
# ──────────────────────────────────────────────────────────────────────

def benchmark_attention_backends(
    model,
    tokenizer,
    prompt: str,
    backends_to_test: Optional[List[str]] = None,
    max_new_tokens: int = 128,
    num_runs: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark different SDPA backends on the same prompt.

    The model must have been loaded with ``attn_implementation="sdpa"``.
    Each backend is activated via the SDPA kernel context manager.

    Returns dict mapping backend name → performance results.
    """
    from src.benchmark_harness import run_single_inference
    from src.vram_monitor import VRAMMonitor, reset_peak_memory

    if backends_to_test is None:
        backends_to_test = ["flash", "mem_efficient", "math"]

    results: Dict[str, Dict[str, Any]] = {}
    monitor = VRAMMonitor()

    for backend in backends_to_test:
        logger.info(f"Testing attention backend: {backend}")

        latencies: List[float] = []
        throughputs: List[float] = []
        peak_vrams: List[float] = []
        errors: List[str] = []

        for run_idx in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                reset_peak_memory()

            try:
                with sdpa_backend(backend):
                    run_result = run_single_inference(
                        model, tokenizer, prompt,
                        max_new_tokens=max_new_tokens,
                    )

                if run_result.oom_error:
                    errors.append(f"OOM on run {run_idx}")
                    break

                if run_result.error_msg:
                    errors.append(run_result.error_msg)
                    break

                latencies.append(run_result.total_time_ms)
                throughputs.append(run_result.tokens_per_second)
                peak_vrams.append(run_result.peak_vram_mb)

            except Exception as e:
                errors.append(str(e))
                logger.warning(f"Backend '{backend}' run {run_idx} failed: {e}")
                break

        if latencies:
            import numpy as np
            results[backend] = {
                "num_runs": len(latencies),
                "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
                "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
                "latency_mean_ms": round(float(np.mean(latencies)), 2),
                "throughput_tok_per_s": round(float(np.mean(throughputs)), 2),
                "peak_vram_mb": round(float(max(peak_vrams)), 1),
                "errors": errors,
                "status": "success",
            }
        else:
            results[backend] = {
                "status": "failed",
                "errors": errors,
            }

        logger.info(f"  -> {backend}: {results[backend]}")

    return results


def benchmark_eager_baseline(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    num_runs: int = 5,
) -> Dict[str, Any]:
    """
    Benchmark the eager (non-SDPA) attention baseline.

    The model must have been loaded with ``attn_implementation="eager"``.
    No kernel context manager is needed since eager is the default path.
    """
    from src.benchmark_harness import run_single_inference
    from src.vram_monitor import reset_peak_memory

    logger.info("Benchmarking eager attention baseline...")

    latencies: List[float] = []
    throughputs: List[float] = []
    peak_vrams: List[float] = []
    errors: List[str] = []

    for run_idx in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            reset_peak_memory()

        try:
            run_result = run_single_inference(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
            )

            if run_result.oom_error:
                errors.append(f"OOM on run {run_idx}")
                break

            latencies.append(run_result.total_time_ms)
            throughputs.append(run_result.tokens_per_second)
            peak_vrams.append(run_result.peak_vram_mb)

        except Exception as e:
            errors.append(str(e))
            logger.warning(f"Eager baseline run {run_idx} failed: {e}")

    if latencies:
        import numpy as np
        return {
            "num_runs": len(latencies),
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "latency_mean_ms": round(float(np.mean(latencies)), 2),
            "throughput_tok_per_s": round(float(np.mean(throughputs)), 2),
            "peak_vram_mb": round(float(max(peak_vrams)), 1),
            "errors": errors,
            "status": "success",
        }
    return {"status": "failed", "errors": errors}
