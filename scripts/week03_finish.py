"""
Week 03 — Finish remaining deliverables (OOM threshold + correctness).
Skips the benchmark sweep which is already saved.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DEFAULT_MODEL_ID, BENCHMARK_CFG, CORRECTNESS_CFG,
    RESULTS_DIR, PROMPTS_DIR,
)
from src.utils import setup_logging, save_results_json, set_seed
from src.model_loader import load_model_4bit
from src.benchmark_harness import find_oom_threshold
from src.correctness import run_correctness_suite

logger = setup_logging("week03")

WEEK_DIR = os.path.join(RESULTS_DIR, "week03_baseline")
os.makedirs(WEEK_DIR, exist_ok=True)


def load_prompts():
    prompt_file = os.path.join(PROMPTS_DIR, "fixed_prompts.json")
    with open(prompt_file) as f:
        return json.load(f)


def run_oom_threshold_search(model, tokenizer):
    logger.info("Searching for OOM threshold...")
    prompts = load_prompts()
    base_prompt = prompts["long_context_seed"]["text"]

    oom_result = find_oom_threshold(
        model=model,
        tokenizer=tokenizer,
        base_prompt=base_prompt,
        max_new_tokens=BENCHMARK_CFG.output_limit,
        start_length=512,
        max_length=4096,
        step=512,
    )

    save_results_json(oom_result, os.path.join(WEEK_DIR, "oom_threshold.json"))
    logger.info(f"  OOM threshold: {oom_result['oom_threshold_tokens']} tokens")
    return oom_result


def run_baseline_correctness(model, tokenizer):
    logger.info("Collecting baseline correctness traces...")
    prompts_data = load_prompts()
    correctness_prompts = [
        p["text"] for p in prompts_data["single_turn"][:CORRECTNESS_CFG.num_fixed_prompts]
    ]

    result = run_correctness_suite(
        model=model,
        tokenizer=tokenizer,
        prompts=correctness_prompts,
        baseline_traces=None,
        num_steps=CORRECTNESS_CFG.num_logit_comparison_steps,
        config_id="baseline",
        seed=BENCHMARK_CFG.seed,
    )

    # Save summary (without full logits)
    traces_for_save = []
    for trace in result["traces"]:
        traces_for_save.append({
            "prompt": trace["prompt"][:100],
            "prompt_tokens": trace["prompt_tokens"],
            "num_actual_steps": trace["num_actual_steps"],
            "token_ids": trace["token_ids"],
            "top1_tokens": trace["top1_tokens"],
        })

    save_results_json(
        {"config_id": "baseline", "traces_summary": traces_for_save},
        os.path.join(WEEK_DIR, "baseline_correctness_summary.json"),
    )

    # Save full traces for later weeks
    save_results_json(result, os.path.join(WEEK_DIR, "baseline_correctness_full.json"))
    logger.info(f"  Saved {len(result['traces'])} correctness traces")
    return result


def main():
    logger.info("=" * 60)
    logger.info("WEEK 03 — Finishing OOM threshold + correctness")
    logger.info("=" * 60)

    set_seed(BENCHMARK_CFG.seed)

    logger.info(f"Loading model: {DEFAULT_MODEL_ID}")
    model, tokenizer = load_model_4bit(DEFAULT_MODEL_ID)

    # 1. OOM threshold
    oom_result = run_oom_threshold_search(model, tokenizer)

    # 2. Correctness traces
    correctness_result = run_baseline_correctness(model, tokenizer)

    # 3. Summary
    summary = {
        "week": "03",
        "title": "Baseline Measurement",
        "status": "COMPLETE",
        "model": DEFAULT_MODEL_ID,
        "oom_threshold_tokens": oom_result.get("oom_threshold_tokens"),
        "correctness_prompts": len(correctness_result.get("traces", [])),
        "deliverables": [
            "baseline_benchmark_results.json",
            "baseline_benchmark_results.csv",
            "oom_threshold.json",
            "baseline_correctness_summary.json",
            "baseline_correctness_full.json",
        ],
    }
    save_results_json(summary, os.path.join(WEEK_DIR, "week03_summary.json"))

    logger.info("=" * 60)
    logger.info("Week 03 COMPLETE. Deliverables saved to: " + WEEK_DIR)
    logger.info("=" * 60)

    import torch
    del model, tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
