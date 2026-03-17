"""
PyTorch Profiler wrapper for identifying bottleneck kernels and layers.

Captures time spent in each CUDA kernel and CPU operator, identifies
the top contributors, and exports traces for manual inspection.
"""

import os
import logging
from typing import Dict, List, Optional, Any

import torch
from torch.profiler import profile, record_function, ProfilerActivity

logger = logging.getLogger("llm_inference.profiler")


def profile_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    num_warmup: int = 2,
    output_dir: Optional[str] = None,
    trace_name: str = "inference_trace",
) -> Dict[str, Any]:
    """
    Profile a single inference pass and identify top kernels.
    
    Args:
        model: The loaded model.
        tokenizer: The tokenizer.
        prompt: Input prompt string.
        max_new_tokens: Tokens to generate.
        num_warmup: Warmup runs before profiling.
        output_dir: Directory to save Chrome trace JSON.
        trace_name: Name for the trace file.
    
    Returns:
        Dictionary with profiling results.
    """
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Warmup
    logger.info(f"Running {num_warmup} warmup iterations...")
    for _ in range(num_warmup):
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Profile
    logger.info("Profiling inference...")
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("model_generate"):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

    # Parse results
    key_averages = prof.key_averages()

    # Top CUDA kernels by time
    cuda_events = sorted(
        [e for e in key_averages if e.device_time_total > 0],
        key=lambda e: e.device_time_total,
        reverse=True,
    )

    # Top CPU operators by time
    cpu_events = sorted(
        [e for e in key_averages if e.cpu_time_total > 0],
        key=lambda e: e.cpu_time_total,
        reverse=True,
    )

    top_cuda = [
        {
            "name": e.key,
            "device_time_ms": round(e.device_time_total / 1000, 3),
            "cpu_time_ms": round(e.cpu_time_total / 1000, 3),
            "calls": e.count,
            "device_memory_mb": round((e.device_memory_usage or 0) / (1024**2), 2),
        }
        for e in cuda_events[:20]
    ]

    top_cpu = [
        {
            "name": e.key,
            "cpu_time_ms": round(e.cpu_time_total / 1000, 3),
            "calls": e.count,
        }
        for e in cpu_events[:20]
    ]

    # Export trace
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        trace_path = os.path.join(output_dir, f"{trace_name}.json")
        prof.export_chrome_trace(trace_path)
        logger.info(f"Trace exported to {trace_path}")

    # Print summary
    total_cuda_ms = sum(e.device_time_total for e in cuda_events) / 1000
    total_cpu_ms = sum(e.cpu_time_total for e in cpu_events) / 1000

    result = {
        "total_cuda_time_ms": round(total_cuda_ms, 2),
        "total_cpu_time_ms": round(total_cpu_ms, 2),
        "top_cuda_kernels": top_cuda,
        "top_cpu_operators": top_cpu,
        "num_tokens_generated": outputs.shape[-1] - inputs["input_ids"].shape[-1],
        "profiler_table": key_averages.table(sort_by="cuda_time_total", row_limit=15),
    }

    return result


def identify_bottlenecks(profile_results: Dict) -> Dict[str, Any]:
    """
    Analyse profiling results and categorise bottleneck kernels.
    
    Groups kernels into categories: attention, linear/matmul,
    activation, normalization, memory, and other.
    """
    kernel_categories = {
        "attention": ["attention", "softmax", "flash", "sdpa", "bmm", "baddbmm"],
        "linear_matmul": ["linear", "matmul", "gemm", "addmm", "mm_"],
        "activation": ["gelu", "relu", "silu", "swiglu", "sigmoid", "tanh"],
        "normalization": ["layer_norm", "rms_norm", "batch_norm", "norm"],
        "memory": ["copy", "memcpy", "memset", "cat", "reshape", "contiguous"],
        "quantization": ["quantize", "dequantize", "bnb", "nf4", "int8"],
    }

    categorised = {cat: [] for cat in kernel_categories}
    categorised["other"] = []

    for kernel in profile_results.get("top_cuda_kernels", []):
        name_lower = kernel["name"].lower()
        matched = False
        for cat, keywords in kernel_categories.items():
            if any(kw in name_lower for kw in keywords):
                categorised[cat].append(kernel)
                matched = True
                break
        if not matched:
            categorised["other"].append(kernel)

    # Compute time per category
    category_summary = {}
    for cat, kernels in categorised.items():
        total_ms = sum(k["device_time_ms"] for k in kernels)
        category_summary[cat] = {
            "total_ms": round(total_ms, 3),
            "num_kernels": len(kernels),
            "top_kernel": kernels[0]["name"] if kernels else None,
        }

    # Sort by time
    category_summary = dict(
        sorted(category_summary.items(), key=lambda x: x[1]["total_ms"], reverse=True)
    )

    total_ms = profile_results.get("total_cuda_time_ms", 1)
    for cat in category_summary:
        category_summary[cat]["pct_of_total"] = round(
            category_summary[cat]["total_ms"] / max(total_ms, 0.001) * 100, 1
        )

    return {
        "category_summary": category_summary,
        "recommendation": _generate_recommendation(category_summary),
    }


def _generate_recommendation(category_summary: Dict) -> str:
    """Generate an optimisation recommendation based on profiling."""
    recommendations = []

    attn = category_summary.get("attention", {})
    if attn.get("pct_of_total", 0) > 30:
        recommendations.append(
            f"Attention takes {attn['pct_of_total']}% of GPU time. "
            "Enabling flash attention or memory-efficient SDPA is high priority."
        )

    mem = category_summary.get("memory", {})
    if mem.get("pct_of_total", 0) > 15:
        recommendations.append(
            f"Memory operations take {mem['pct_of_total']}% of GPU time. "
            "Consider reducing data movement with in-place operations or better "
            "KV-cache management."
        )

    quant = category_summary.get("quantization", {})
    if quant.get("pct_of_total", 0) > 10:
        recommendations.append(
            f"Quantization kernels take {quant['pct_of_total']}% of GPU time. "
            "This overhead is expected with 4-bit weights; monitor for regressions."
        )

    if not recommendations:
        recommendations.append(
            "No single category dominates. Optimisation should focus on the "
            "top individual kernels."
        )

    return " ".join(recommendations)
