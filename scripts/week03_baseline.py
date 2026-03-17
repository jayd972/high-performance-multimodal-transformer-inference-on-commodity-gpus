"""
Week 03 — Baseline Measurement

Deliverables:
  - Baseline results dataset (CSV/JSON) with latency, throughput, memory, OOM thresholds
  - Correctness harness code with stored baseline logits/token traces
  - Baseline plots showing scaling with sequence length
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DEFAULT_MODEL_ID, BENCHMARK_CFG, CORRECTNESS_CFG,
    RESULTS_DIR, PROMPTS_DIR,
)
from src.utils import setup_logging, save_results_json, save_results_csv, set_seed
from src.model_loader import load_model_4bit
from src.benchmark_harness import (
    run_benchmark_sweep, find_oom_threshold, results_to_table,
    prepare_prompt_at_length,
)
from src.correctness import run_correctness_suite

logger = setup_logging("week03")

WEEK_DIR = os.path.join(RESULTS_DIR, "week03_baseline")
os.makedirs(WEEK_DIR, exist_ok=True)


def load_prompts():
    """Load fixed prompts."""
    prompt_file = os.path.join(PROMPTS_DIR, "fixed_prompts.json")
    with open(prompt_file) as f:
        return json.load(f)


def run_baseline_benchmarks(model, tokenizer):
    """Run the full baseline benchmark suite."""
    logger.info("Running baseline benchmark sweep...")

    prompts = load_prompts()
    base_prompt = prompts["single_turn"][0]["text"]

    results = run_benchmark_sweep(
        model=model,
        tokenizer=tokenizer,
        base_prompt=base_prompt,
        config_id="baseline",
        prompt_lengths=BENCHMARK_CFG.prompt_lengths,
        output_limit=BENCHMARK_CFG.output_limit,
        batch_sizes=BENCHMARK_CFG.batch_sizes,
        num_warmup=BENCHMARK_CFG.num_warmup_runs,
        num_runs=BENCHMARK_CFG.num_benchmark_runs,
        seed=BENCHMARK_CFG.seed,
    )

    # Save as JSON
    results_data = [
        {
            "config_id": r.config_id,
            "prompt_length": r.prompt_length,
            "batch_size": r.batch_size,
            "output_limit": r.output_limit,
            "num_runs": r.num_runs,
            "latency_p50_ms": r.latency_p50_ms,
            "latency_p95_ms": r.latency_p95_ms,
            "latency_mean_ms": r.latency_mean_ms,
            "latency_std_ms": r.latency_std_ms,
            "throughput_tok_per_s": r.throughput_tokens_per_s,
            "peak_vram_mb": r.peak_vram_mb,
            "peak_cpu_ram_mb": r.peak_cpu_ram_mb,
            "oom_occurred": r.oom_occurred,
        }
        for r in results
    ]

    save_results_json(
        results_data,
        os.path.join(WEEK_DIR, "baseline_benchmark_results.json"),
    )

    # Save as CSV
    table_rows = results_to_table(results)
    save_results_csv(table_rows, os.path.join(WEEK_DIR, "baseline_benchmark_results.csv"))

    return results


def run_oom_threshold_search(model, tokenizer):
    """Search for OOM threshold."""
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

    save_results_json(
        oom_result,
        os.path.join(WEEK_DIR, "oom_threshold.json"),
    )

    logger.info(f"  OOM threshold: {oom_result['oom_threshold_tokens']} tokens")
    return oom_result


def run_baseline_correctness(model, tokenizer):
    """Collect baseline correctness traces."""
    logger.info("Collecting baseline correctness traces...")

    prompts_data = load_prompts()
    correctness_prompts = [
        p["text"] for p in prompts_data["single_turn"][:CORRECTNESS_CFG.num_fixed_prompts]
    ]

    result = run_correctness_suite(
        model=model,
        tokenizer=tokenizer,
        prompts=correctness_prompts,
        baseline_traces=None,  # This IS the baseline
        num_steps=CORRECTNESS_CFG.num_logit_comparison_steps,
        config_id="baseline",
        seed=BENCHMARK_CFG.seed,
    )

    # Save traces (without full logits to save space)
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

    # Save full traces (with logits) for comparison in later weeks
    save_results_json(
        result,
        os.path.join(WEEK_DIR, "baseline_correctness_full.json"),
    )

    return result


def main():
    logger.info("=" * 60)
    logger.info("WEEK 03 — Baseline Measurement")
    logger.info("=" * 60)

    set_seed(BENCHMARK_CFG.seed)

    # Load model
    logger.info(f"Loading model: {DEFAULT_MODEL_ID}")
    model, tokenizer = load_model_4bit(DEFAULT_MODEL_ID)

    # 1. Baseline benchmarks
    bench_results = run_baseline_benchmarks(model, tokenizer)

    # 2. OOM threshold
    oom_result = run_oom_threshold_search(model, tokenizer)

    # 3. Correctness traces
    correctness_result = run_baseline_correctness(model, tokenizer)

    # Summary
    summary = {
        "week": "03",
        "title": "Baseline Measurement",
        "status": "COMPLETE",
        "model": DEFAULT_MODEL_ID,
        "num_benchmark_configs": len(bench_results),
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
    logger.info("Week 03 deliverables saved to: " + WEEK_DIR)
    logger.info("=" * 60)

    # Clean up
    import torch
    del model, tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
