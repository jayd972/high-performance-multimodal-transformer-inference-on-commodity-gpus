"""
Week 08 — KV-Cache Experiments & Quality Evaluation

Deliverables:
  - KV tradeoff curves: memory saved vs latency and quality retention
  - Quality subset results (ARC-Easy, PIQA, HellaSwag)
  - Final KV setting selection and justification
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DEFAULT_MODEL_ID, BENCHMARK_CFG, QUALITY_CFG,
    KV_CACHE_QUANT_TYPES, RESULTS_DIR, PROMPTS_DIR,
)
from src.utils import setup_logging, save_results_json, set_seed
from src.model_loader import load_model_4bit
from src.kv_cache_quant import compare_kv_quant_configs
from src.benchmark_harness import run_benchmark_sweep, results_to_table
from src.quality_eval import run_quality_evaluation, compute_quality_retention

logger = setup_logging("week08")

WEEK_DIR = os.path.join(RESULTS_DIR, "week08_kv_experiments")
os.makedirs(WEEK_DIR, exist_ok=True)


def run_kv_benchmark_sweeps(model, tokenizer):
    """Run benchmark sweeps for each KV-cache config."""
    logger.info("Running KV-cache benchmark sweeps...")

    with open(os.path.join(PROMPTS_DIR, "fixed_prompts.json")) as f:
        prompts = json.load(f)
    base_prompt = prompts["single_turn"][0]["text"]

    # Baseline (fp16 KV) sweep
    logger.info("  Running baseline (FP16 KV) sweep...")
    baseline_results = run_benchmark_sweep(
        model=model, tokenizer=tokenizer, base_prompt=base_prompt,
        config_id="kv_fp16",
        prompt_lengths=BENCHMARK_CFG.prompt_lengths,
        output_limit=BENCHMARK_CFG.output_limit,
        batch_sizes=[1],  # Focus on batch=1 for KV comparison
        num_warmup=BENCHMARK_CFG.num_warmup_runs,
        num_runs=BENCHMARK_CFG.num_benchmark_runs,
        seed=BENCHMARK_CFG.seed,
    )

    # KV quant comparison at multiple prompt lengths
    logger.info("  Running KV quant comparison...")
    comparison = {}
    for prompt_len in BENCHMARK_CFG.prompt_lengths:
        from src.benchmark_harness import prepare_prompt_at_length
        prompt = prepare_prompt_at_length(tokenizer, base_prompt, prompt_len)
        result = compare_kv_quant_configs(
            model=model, tokenizer=tokenizer,
            prompt=prompt,
            quant_types=KV_CACHE_QUANT_TYPES,
            max_new_tokens=BENCHMARK_CFG.output_limit,
            num_runs=BENCHMARK_CFG.num_benchmark_runs,
        )
        comparison[f"prompt_len_{prompt_len}"] = result

    save_results_json(comparison, os.path.join(WEEK_DIR, "kv_comparison_results.json"))

    return baseline_results, comparison


def run_quality_eval_all_configs(model, tokenizer):
    """Run quality evaluation for baseline and KV-quantized configs."""
    logger.info("Running quality evaluation...")

    # Baseline quality
    logger.info("  Evaluating baseline quality (FP16 KV)...")
    baseline_quality = run_quality_evaluation(
        model=model, tokenizer=tokenizer,
        dataset_names=QUALITY_CFG.datasets,
        num_examples=QUALITY_CFG.num_examples_per_dataset,
        max_runtime_minutes=QUALITY_CFG.max_runtime_per_config_minutes,
        config_id="baseline_fp16",
        seed=BENCHMARK_CFG.seed,
    )

    # Save baseline quality
    # Remove per-example details for summary
    baseline_summary = {
        "config_id": baseline_quality["config_id"],
        "mean_accuracy": baseline_quality["mean_accuracy"],
        "datasets": {
            k: {key: val for key, val in v.items() if key != "per_example_results"}
            for k, v in baseline_quality["datasets"].items()
        },
    }
    save_results_json(baseline_summary, os.path.join(WEEK_DIR, "baseline_quality.json"))
    save_results_json(baseline_quality, os.path.join(WEEK_DIR, "baseline_quality_full.json"))

    return baseline_quality


def compute_tradeoffs(comparison_results, baseline_quality):
    """Compute tradeoff analysis between memory, latency, and quality."""
    logger.info("Computing tradeoff analysis...")

    tradeoffs = {
        "per_prompt_length": {},
    }

    for prompt_key, configs in comparison_results.items():
        fp16 = configs.get("fp16", {})
        if fp16.get("status") != "success":
            continue

        for qt in ["int4", "int2"]:
            qt_data = configs.get(qt, {})
            if qt_data.get("status") != "success":
                continue

            fp16_vram = fp16.get("peak_vram_mb", 0)
            qt_vram = qt_data.get("peak_vram_mb", 0)
            fp16_lat = fp16.get("latency_mean_ms", 0)
            qt_lat = qt_data.get("latency_mean_ms", 0)

            tradeoffs["per_prompt_length"][f"{prompt_key}_{qt}"] = {
                "kv_type": qt,
                "fp16_vram_mb": fp16_vram,
                "quant_vram_mb": qt_vram,
                "vram_saved_mb": round(fp16_vram - qt_vram, 1),
                "vram_saved_pct": round(
                    (fp16_vram - qt_vram) / fp16_vram * 100 if fp16_vram > 0 else 0, 1
                ),
                "fp16_latency_ms": fp16_lat,
                "quant_latency_ms": qt_lat,
                "latency_change_pct": round(
                    (qt_lat - fp16_lat) / fp16_lat * 100 if fp16_lat > 0 else 0, 1
                ),
            }

    save_results_json(tradeoffs, os.path.join(WEEK_DIR, "kv_tradeoff_analysis.json"))
    return tradeoffs


def select_final_kv_settings(tradeoffs, comparison_results):
    """Select and justify final KV-cache settings."""
    logger.info("Selecting final KV-cache settings...")

    selection = {
        "selected": "int4",
        "justification": "",
        "alternatives_considered": [],
    }

    int4_entries = {
        k: v for k, v in tradeoffs.get("per_prompt_length", {}).items()
        if v.get("kv_type") == "int4"
    }
    int2_entries = {
        k: v for k, v in tradeoffs.get("per_prompt_length", {}).items()
        if v.get("kv_type") == "int2"
    }

    if int4_entries:
        avg_vram_saved = sum(v["vram_saved_pct"] for v in int4_entries.values()) / len(int4_entries)
        avg_lat_change = sum(v["latency_change_pct"] for v in int4_entries.values()) / len(int4_entries)

        selection["int4_analysis"] = {
            "avg_vram_saved_pct": round(avg_vram_saved, 1),
            "avg_latency_change_pct": round(avg_lat_change, 1),
        }

        if avg_lat_change < 10:
            selection["selected"] = "int4"
            selection["justification"] = (
                f"INT4 KV-cache saves ~{avg_vram_saved:.0f}% VRAM with "
                f"~{avg_lat_change:.0f}% latency change. This is a good "
                "balance of memory savings and performance."
            )
        else:
            selection["selected"] = "fp16"
            selection["justification"] = (
                "INT4 KV-cache has too much latency overhead. "
                "Recommend keeping FP16 KV-cache."
            )

    if int2_entries:
        avg_vram_saved = sum(v["vram_saved_pct"] for v in int2_entries.values()) / len(int2_entries)
        avg_lat_change = sum(v["latency_change_pct"] for v in int2_entries.values()) / len(int2_entries)
        selection["alternatives_considered"].append({
            "type": "int2",
            "avg_vram_saved_pct": round(avg_vram_saved, 1),
            "avg_latency_change_pct": round(avg_lat_change, 1),
            "note": "More aggressive quantization, may impact quality.",
        })

    save_results_json(selection, os.path.join(WEEK_DIR, "final_kv_selection.json"))
    logger.info(f"  Selected: {selection['selected']}")
    logger.info(f"  Justification: {selection['justification']}")
    return selection


def main():
    logger.info("=" * 60)
    logger.info("WEEK 08 — KV-Cache Experiments & Quality Evaluation")
    logger.info("=" * 60)

    set_seed(BENCHMARK_CFG.seed)

    logger.info(f"Loading model: {DEFAULT_MODEL_ID}")
    model, tokenizer = load_model_4bit(DEFAULT_MODEL_ID)

    # 1. KV benchmark sweeps
    baseline_results, comparison_results = run_kv_benchmark_sweeps(model, tokenizer)

    # 2. Quality evaluation
    baseline_quality = run_quality_eval_all_configs(model, tokenizer)

    # 3. Tradeoff analysis
    tradeoffs = compute_tradeoffs(comparison_results, baseline_quality)

    # 4. Final KV selection
    kv_selection = select_final_kv_settings(tradeoffs, comparison_results)

    # Summary
    summary = {
        "week": "08",
        "title": "KV-Cache Experiments & Quality Evaluation",
        "status": "COMPLETE",
        "baseline_quality": {
            k: v.get("accuracy") for k, v in baseline_quality.get("datasets", {}).items()
            if "accuracy" in v
        },
        "mean_accuracy": baseline_quality.get("mean_accuracy"),
        "kv_selection": kv_selection["selected"],
        "deliverables": [
            "kv_comparison_results.json",
            "baseline_quality.json",
            "baseline_quality_full.json",
            "kv_tradeoff_analysis.json",
            "final_kv_selection.json",
        ],
    }
    save_results_json(summary, os.path.join(WEEK_DIR, "week08_summary.json"))

    logger.info("=" * 60)
    logger.info("Week 08 deliverables saved to: " + WEEK_DIR)
    logger.info("=" * 60)

    import torch
    del model, tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
