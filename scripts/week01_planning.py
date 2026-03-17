"""
Week 01 — Planning & Scope Definition

Deliverables:
  - Model shortlist and selection criteria
  - VRAM budget worksheet with headroom target
  - Fixed benchmark settings document
  - Fixed prompt set v1 and evaluation protocol document
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODEL_CANDIDATES, DEFAULT_MODEL_KEY, DEFAULT_MODEL_ID,
    VRAM_TOTAL_GB, VRAM_WEIGHT_BUDGET_GB, VRAM_HEADROOM_GB,
    BENCHMARK_CFG, CORRECTNESS_CFG, QUALITY_CFG,
    WORKLOAD_TYPES, CONFIGURATION_IDS, RESULTS_DIR, PROMPTS_DIR,
)
from src.utils import save_results_json, setup_logging

logger = setup_logging("week01")

WEEK_DIR = os.path.join(RESULTS_DIR, "week01_planning")
os.makedirs(WEEK_DIR, exist_ok=True)


def generate_model_shortlist():
    """Document model candidates and selection criteria."""
    shortlist = {
        "selection_criteria": {
            "parameter_range": "2B–4B parameters",
            "vram_constraint": f"4-bit weights must fit in ≤{VRAM_WEIGHT_BUDGET_GB} GB",
            "requirements": [
                "Open-source with permissive license",
                "Good quality-to-size ratio",
                "Compatible with bitsandbytes 4-bit quantization",
                "Active community and well-tested",
            ],
        },
        "candidates": MODEL_CANDIDATES,
        "primary_selection": {
            "key": DEFAULT_MODEL_KEY,
            "model_id": DEFAULT_MODEL_ID,
            "rationale": (
                "Qwen2.5-3B-Instruct offers the best quality-to-size ratio in the "
                "2B-4B range, has strong instruction following, and is well tested "
                "with bitsandbytes 4-bit quantization. Estimated 4-bit weight "
                "footprint of ~1.8 GB leaves ample headroom for KV-cache and "
                "runtime buffers."
            ),
        },
        "fallback_options": [
            {
                "key": "phi-2",
                "rationale": "Smaller footprint, MIT license, fallback if Qwen has issues.",
            },
            {
                "key": "gemma-2-2b",
                "rationale": "Most compact option, good for worst-case VRAM scenarios.",
            },
        ],
    }
    return shortlist


def generate_vram_budget():
    """Document VRAM budget and headroom analysis."""
    budget = {
        "gpu_specs": {
            "target_gpu": "NVIDIA RTX 3050 Ti Laptop (4 GB VRAM)",
            "total_vram_gb": VRAM_TOTAL_GB,
            "compute_capability": "8.6 (Ampere)",
        },
        "budget_allocation": {
            "weight_budget_gb": VRAM_WEIGHT_BUDGET_GB,
            "headroom_gb": VRAM_HEADROOM_GB,
            "headroom_breakdown": {
                "kv_cache_gb": 0.4,
                "activation_buffers_gb": 0.2,
                "pytorch_overhead_gb": 0.2,
            },
        },
        "model_fit_analysis": {},
    }

    for key, info in MODEL_CANDIDATES.items():
        fits = info["est_4bit_gb"] <= VRAM_WEIGHT_BUDGET_GB
        budget["model_fit_analysis"][key] = {
            "estimated_4bit_gb": info["est_4bit_gb"],
            "fits_budget": fits,
            "remaining_headroom_gb": round(VRAM_WEIGHT_BUDGET_GB - info["est_4bit_gb"], 2),
        }

    return budget


def generate_benchmark_settings():
    """Document fixed benchmark settings."""
    return {
        "prompt_lengths_tokens": BENCHMARK_CFG.prompt_lengths,
        "output_limit_tokens": BENCHMARK_CFG.output_limit,
        "batch_sizes": BENCHMARK_CFG.batch_sizes,
        "decoding": {
            "strategy": "greedy",
            "temperature": BENCHMARK_CFG.temperature,
            "do_sample": BENCHMARK_CFG.do_sample,
            "top_p": BENCHMARK_CFG.top_p,
        },
        "warmup_runs": BENCHMARK_CFG.num_warmup_runs,
        "benchmark_runs": BENCHMARK_CFG.num_benchmark_runs,
        "seed": BENCHMARK_CFG.seed,
        "workload_types": WORKLOAD_TYPES,
        "metrics_collected": [
            "p50 latency (ms)",
            "p95 latency (ms)",
            "tokens/second throughput",
            "peak VRAM (MB)",
            "peak CPU RAM (MB)",
            "out-of-memory threshold (tokens)",
        ],
    }


def generate_evaluation_protocol():
    """Document correctness and quality evaluation protocol."""
    return {
        "correctness": {
            "method": "Deterministic greedy decoding comparison",
            "metrics": [
                "Top-1 token agreement rate",
                "Maximum absolute logit difference",
            ],
            "num_steps": CORRECTNESS_CFG.num_logit_comparison_steps,
            "tolerance": CORRECTNESS_CFG.max_logit_diff_tolerance,
            "num_prompts": CORRECTNESS_CFG.num_fixed_prompts,
            "seed": BENCHMARK_CFG.seed,
        },
        "quality": {
            "method": "0-shot log-likelihood evaluation",
            "datasets": list(zip(QUALITY_CFG.datasets, QUALITY_CFG.dataset_display_names)),
            "num_examples_per_dataset": QUALITY_CFG.num_examples_per_dataset,
            "max_runtime_per_config_minutes": QUALITY_CFG.max_runtime_per_config_minutes,
            "metric": "Accuracy (% correct)",
            "quality_drop_threshold": "≤2% accuracy drop considered acceptable",
        },
        "configuration_ids": CONFIGURATION_IDS,
    }


def main():
    logger.info("=" * 60)
    logger.info("WEEK 01 — Planning & Scope Definition")
    logger.info("=" * 60)

    # 1. Model shortlist
    logger.info("Generating model shortlist...")
    shortlist = generate_model_shortlist()
    save_results_json(shortlist, os.path.join(WEEK_DIR, "model_shortlist.json"))
    logger.info(f"  Primary model: {shortlist['primary_selection']['model_id']}")

    # 2. VRAM budget
    logger.info("Generating VRAM budget worksheet...")
    budget = generate_vram_budget()
    save_results_json(budget, os.path.join(WEEK_DIR, "vram_budget.json"))
    logger.info(f"  Weight budget: {VRAM_WEIGHT_BUDGET_GB} GB / {VRAM_TOTAL_GB} GB total")

    # 3. Benchmark settings
    logger.info("Documenting benchmark settings...")
    settings = generate_benchmark_settings()
    save_results_json(settings, os.path.join(WEEK_DIR, "benchmark_settings.json"))
    logger.info(f"  Prompt lengths: {settings['prompt_lengths_tokens']}")
    logger.info(f"  Batch sizes: {settings['batch_sizes']}")

    # 4. Evaluation protocol
    logger.info("Documenting evaluation protocol...")
    protocol = generate_evaluation_protocol()
    save_results_json(protocol, os.path.join(WEEK_DIR, "evaluation_protocol.json"))

    # 5. Verify prompt set exists
    prompt_file = os.path.join(PROMPTS_DIR, "fixed_prompts.json")
    if os.path.exists(prompt_file):
        with open(prompt_file) as f:
            prompts = json.load(f)
        logger.info(
            f"  Prompt set: {len(prompts['single_turn'])} single-turn, "
            f"{len(prompts['multi_turn'])} multi-turn"
        )
    else:
        logger.warning("  Fixed prompt set not found!")

    # Summary
    summary = {
        "week": "01",
        "title": "Planning & Scope Definition",
        "status": "COMPLETE",
        "deliverables": [
            "model_shortlist.json",
            "vram_budget.json",
            "benchmark_settings.json",
            "evaluation_protocol.json",
        ],
        "primary_model": DEFAULT_MODEL_ID,
        "vram_budget_gb": VRAM_WEIGHT_BUDGET_GB,
    }
    save_results_json(summary, os.path.join(WEEK_DIR, "week01_summary.json"))

    logger.info("=" * 60)
    logger.info("Week 01 deliverables saved to: " + WEEK_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
