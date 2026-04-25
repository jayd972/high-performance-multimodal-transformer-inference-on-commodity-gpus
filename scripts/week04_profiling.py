"""
Week 04 — Profiling & Attention Backend Planning

Deliverables:
  - Bottleneck summary with top kernels and layers
  - Attention backend plan with primary and fallback paths
  - Proof logging: backend selection logs and profiler traces
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_MODEL_ID, MODEL_CANDIDATES, BENCHMARK_CFG, RESULTS_DIR, PROMPTS_DIR
from src.utils import setup_logging, save_results_json, set_seed
from src.model_loader import load_model_4bit
from src.profiler import profile_inference, identify_bottlenecks
from src.attention_backends import check_attention_backends

logger = setup_logging("week04")

WEEK_DIR = os.path.join(RESULTS_DIR, "week04_profiling")
os.makedirs(WEEK_DIR, exist_ok=True)


def run_profiling(model, tokenizer):
    """Profile baseline inference and identify bottlenecks."""
    logger.info("Profiling baseline inference...")

    # Load a medium-length prompt
    with open(os.path.join(PROMPTS_DIR, "fixed_prompts.json")) as f:
        prompts = json.load(f)
    prompt = prompts["single_turn"][0]["text"]

    # Profile
    profile_results = profile_inference(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=64,
        num_warmup=2,
        output_dir=WEEK_DIR,
        trace_name="baseline_profile_trace",
    )

    save_results_json(
        {
            "total_cuda_time_ms": profile_results["total_cuda_time_ms"],
            "total_cpu_time_ms": profile_results["total_cpu_time_ms"],
            "top_cuda_kernels": profile_results["top_cuda_kernels"],
            "top_cpu_operators": profile_results["top_cpu_operators"],
            "num_tokens_generated": profile_results["num_tokens_generated"],
        },
        os.path.join(WEEK_DIR, "profiling_results.json"),
    )

    # Save the profiler table as text
    with open(os.path.join(WEEK_DIR, "profiler_table.txt"), "w") as f:
        f.write(profile_results.get("profiler_table", "No table available"))

    return profile_results


def analyse_bottlenecks(profile_results):
    """Analyse and categorise bottleneck kernels."""
    logger.info("Analysing bottlenecks...")

    analysis = identify_bottlenecks(profile_results)

    save_results_json(analysis, os.path.join(WEEK_DIR, "bottleneck_analysis.json"))

    # Log summary
    for cat, info in analysis["category_summary"].items():
        logger.info(
            f"  {cat}: {info['total_ms']:.1f}ms "
            f"({info['pct_of_total']:.1f}%)"
        )

    logger.info(f"  Recommendation: {analysis['recommendation']}")
    return analysis


def plan_attention_backends():
    """Check and document attention backend availability."""
    logger.info("Checking attention backend availability...")

    backend_info = check_attention_backends()

    # Build plan
    plan = {
        "hardware_info": {
            "gpu": backend_info.get("gpu_name", "Unknown"),
            "compute_capability": backend_info.get("compute_capability", "Unknown"),
            "torch_version": backend_info.get("torch_version", "Unknown"),
        },
        "backend_availability": {
            "sdpa_available": backend_info.get("sdpa_available", False),
            "flash_sdp": backend_info.get("flash_sdp_available", False),
            "mem_efficient_sdp": backend_info.get("mem_efficient_sdp_available", False),
            "math_sdp": backend_info.get("math_sdp_available", True),
            "flash_attention_2": backend_info.get("flash_attn_2_available", False),
        },
        "plan": {},
    }

    # Determine primary and fallback
    if backend_info.get("flash_sdp_available"):
        plan["plan"]["primary"] = "sdpa_flash"
        plan["plan"]["primary_notes"] = (
            "Flash SDP available via PyTorch SDPA. This is the preferred "
            "attention backend for memory efficiency."
        )
    elif backend_info.get("mem_efficient_sdp_available"):
        plan["plan"]["primary"] = "sdpa_mem_efficient"
        plan["plan"]["primary_notes"] = (
            "Flash SDP not available. Using memory-efficient SDP as primary. "
            "This still avoids materializing the full attention matrix."
        )
    else:
        plan["plan"]["primary"] = "sdpa_math"
        plan["plan"]["primary_notes"] = (
            "Only math SDP available. Will benchmark against eager baseline."
        )

    plan["plan"]["fallback"] = "sdpa_math" if plan["plan"]["primary"] != "sdpa_math" else "eager"
    plan["plan"]["fallback_notes"] = "Math SDP or eager attention as fallback."

    plan["plan"]["backends_to_benchmark"] = ["eager"]
    for backend in ["sdpa_flash", "sdpa_mem_efficient", "sdpa_math"]:
        key = backend.replace("sdpa_", "") + "_sdp_available"
        alt_key = backend.replace("sdpa_", "") + "_sdp"
        if backend_info.get(key, False) or backend_info.get(alt_key + "_available", False):
            plan["plan"]["backends_to_benchmark"].append(backend)

    save_results_json(plan, os.path.join(WEEK_DIR, "attention_backend_plan.json"))

    logger.info(f"  Primary backend: {plan['plan']['primary']}")
    logger.info(f"  Fallback backend: {plan['plan']['fallback']}")
    logger.info(f"  Backends to benchmark: {plan['plan']['backends_to_benchmark']}")

    return plan


def main():
    logger.info("=" * 60)
    logger.info("WEEK 04 — Profiling & Attention Backend Planning")
    logger.info("=" * 60)

    set_seed(BENCHMARK_CFG.seed)

    # Load model with SDPA
    logger.info(f"Loading model with SDPA: {DEFAULT_MODEL_ID}")
    model, tokenizer = load_model_4bit(
        DEFAULT_MODEL_ID, attn_implementation="sdpa"
    )

    # 1. Profile
    profile_results = run_profiling(model, tokenizer)

    # 2. Bottleneck analysis
    bottleneck_analysis = analyse_bottlenecks(profile_results)

    # Clean up model before backend checks
    import torch
    del model, tokenizer
    torch.cuda.empty_cache()

    # 3. Attention backend plan
    backend_plan = plan_attention_backends()

    # Summary
    summary = {
        "week": "04",
        "title": "Profiling & Attention Backend Planning",
        "status": "COMPLETE",
        "top_bottleneck_category": list(bottleneck_analysis["category_summary"].keys())[0],
        "primary_attention_backend": backend_plan["plan"]["primary"],
        "recommendation": bottleneck_analysis["recommendation"],
        "deliverables": [
            "profiling_results.json",
            "profiler_table.txt",
            "baseline_profile_trace.json",
            "bottleneck_analysis.json",
            "attention_backend_plan.json",
        ],
    }
    save_results_json(summary, os.path.join(WEEK_DIR, "week04_summary.json"))

    logger.info("=" * 60)
    logger.info("Week 04 deliverables saved to: " + WEEK_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs="+", default=None,
                        help="Model key(s) — week04 uses DEFAULT_MODEL_ID regardless")
    parser.parse_args()
    main()
