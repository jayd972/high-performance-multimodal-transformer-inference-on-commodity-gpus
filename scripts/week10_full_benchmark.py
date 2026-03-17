"""
Week 10 — Full Benchmark Suite

Deliverables:
  - Final benchmark dataset and consolidated summary table
  - Final plots ready for report
  - Draft configuration guideline section
"""

import os
import sys
import json
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_MODEL_ID, BENCHMARK_CFG, RESULTS_DIR, PROMPTS_DIR
from src.utils import (
    setup_logging, save_results_json, save_results_csv,
    set_seed, load_results_json,
)
from src.model_loader import load_model_4bit
from src.attention_backends import sdpa_backend
from src.benchmark_harness import (
    run_benchmark_sweep, find_oom_threshold, results_to_table,
)
from src.kv_cache_quant import run_inference_with_kv_quant
import torch

logger = setup_logging("week10")

WEEK_DIR = os.path.join(RESULTS_DIR, "week10_full_benchmark")
os.makedirs(WEEK_DIR, exist_ok=True)


def load_runbook():
    """Load the final benchmark runbook from week 09."""
    runbook_path = os.path.join(
        RESULTS_DIR, "week09_combined", "final_benchmark_runbook.json"
    )
    if os.path.exists(runbook_path):
        return load_results_json(runbook_path)
    else:
        logger.warning("Runbook not found. Using default configurations.")
        return {
            "configurations": [
                {"id": "baseline", "attn_implementation": "eager", "kv_quant": "fp16"},
                {"id": "sdpa_default", "attn_implementation": "sdpa", "kv_quant": "fp16"},
                {"id": "kv_int4", "attn_implementation": "eager", "kv_quant": "int4"},
            ],
        }


def run_full_benchmark_suite():
    """Run benchmarks for every configuration in the runbook."""
    logger.info("Running full benchmark suite...")

    runbook = load_runbook()
    configs = runbook.get("configurations", [])

    with open(os.path.join(PROMPTS_DIR, "fixed_prompts.json")) as f:
        prompts = json.load(f)
    base_prompt = prompts["single_turn"][0]["text"]
    long_prompt = prompts["long_context_seed"]["text"]

    all_results = []
    oom_thresholds = {}

    for config in configs:
        config_id = config["id"]
        attn_impl = config.get("attn_implementation", "eager")
        sdpa_bk = config.get("sdpa_backend", None)
        kv_quant = config.get("kv_quant", "fp16")

        logger.info(f"\n{'='*40}")
        logger.info(f"Configuration: {config_id}")
        logger.info(f"  attn={attn_impl}, sdpa_backend={sdpa_bk}, kv_quant={kv_quant}")
        logger.info(f"{'='*40}")

        # Load model with appropriate attention
        torch.cuda.empty_cache()
        gc.collect()

        try:
            model, tokenizer = load_model_4bit(
                DEFAULT_MODEL_ID, attn_implementation=attn_impl,
            )
        except Exception as e:
            logger.error(f"Failed to load model for {config_id}: {e}")
            continue

        # Run benchmark sweep
        try:
            ctx = sdpa_backend(sdpa_bk) if sdpa_bk else sdpa_backend("default")
            with ctx:
                if kv_quant == "fp16":
                    results = run_benchmark_sweep(
                        model=model, tokenizer=tokenizer,
                        base_prompt=base_prompt,
                        config_id=config_id,
                        prompt_lengths=BENCHMARK_CFG.prompt_lengths,
                        output_limit=BENCHMARK_CFG.output_limit,
                        batch_sizes=BENCHMARK_CFG.batch_sizes,
                        num_warmup=BENCHMARK_CFG.num_warmup_runs,
                        num_runs=BENCHMARK_CFG.num_benchmark_runs,
                        seed=BENCHMARK_CFG.seed,
                    )
                    all_results.extend(results)
                else:
                    from src.kv_cache_quant import compare_kv_quant_configs
                    from src.benchmark_harness import prepare_prompt_at_length, BenchmarkResult
                    import numpy as np

                    for prompt_len in BENCHMARK_CFG.prompt_lengths:
                        prompt = prepare_prompt_at_length(tokenizer, base_prompt, prompt_len)
                        kv_result = compare_kv_quant_configs(
                            model=model, tokenizer=tokenizer,
                            prompt=prompt,
                            quant_types=[kv_quant],
                            max_new_tokens=BENCHMARK_CFG.output_limit,
                            num_runs=BENCHMARK_CFG.num_benchmark_runs,
                        )
                        qt_data = kv_result.get(kv_quant, {})
                        bench = BenchmarkResult(
                            config_id=config_id,
                            prompt_length=prompt_len,
                            output_limit=BENCHMARK_CFG.output_limit,
                            batch_size=1,
                            num_runs=qt_data.get("num_runs", 0),
                            latency_p50_ms=qt_data.get("latency_p50_ms", 0),
                            latency_p95_ms=qt_data.get("latency_p50_ms", 0),
                            latency_mean_ms=qt_data.get("latency_mean_ms", 0),
                            throughput_tokens_per_s=round(
                                BENCHMARK_CFG.output_limit / (qt_data.get("latency_mean_ms", 1) / 1000), 2
                            ) if qt_data.get("latency_mean_ms", 0) > 0 else 0,
                            peak_vram_mb=qt_data.get("peak_vram_mb", 0),
                            oom_occurred=qt_data.get("status") != "success",
                        )
                        all_results.append(bench)

                # OOM threshold
                logger.info(f"  Finding OOM threshold for {config_id}...")
                oom = find_oom_threshold(
                    model=model, tokenizer=tokenizer,
                    base_prompt=long_prompt,
                    max_new_tokens=BENCHMARK_CFG.output_limit,
                    start_length=128, max_length=8192, step=256,
                )
                oom_thresholds[config_id] = oom

        except Exception as e:
            logger.error(f"Benchmark failed for {config_id}: {e}")

        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()

    return all_results, oom_thresholds


def generate_consolidated_table(all_results):
    """Generate the final consolidated benchmark table."""
    table_rows = results_to_table(all_results)
    save_results_csv(table_rows, os.path.join(WEEK_DIR, "final_benchmark_table.csv"))
    save_results_json(table_rows, os.path.join(WEEK_DIR, "final_benchmark_table.json"))
    logger.info(f"Consolidated table: {len(table_rows)} rows")
    return table_rows


def generate_configuration_guidelines(all_results, oom_thresholds):
    """Generate practical configuration guidelines."""

    guidelines = {
        "title": "Configuration Guidelines for 4 GB VRAM Laptops",
        "target_hardware": "NVIDIA RTX 3050 Ti Laptop (4 GB VRAM)",
        "model": DEFAULT_MODEL_ID,
        "quantization": "4-bit NF4 with double quantization",
        "recommendations": [],
        "oom_thresholds": {
            k: v.get("oom_threshold_tokens", "N/A")
            for k, v in oom_thresholds.items()
        },
    }

    # Analyze results to generate recommendations
    config_summaries = {}
    for r in all_results:
        cid = r.config_id
        if cid not in config_summaries:
            config_summaries[cid] = []
        config_summaries[cid].append(r)

    # Find best config for each metric
    best_latency = None
    best_throughput = None
    best_vram = None

    for cid, results in config_summaries.items():
        # Average across prompt lengths for batch_size=1
        bs1 = [r for r in results if r.batch_size == 1 and not r.oom_occurred]
        if not bs1:
            continue

        avg_p95 = sum(r.latency_p95_ms for r in bs1) / len(bs1)
        avg_tps = sum(r.throughput_tokens_per_s for r in bs1) / len(bs1)
        avg_vram = max(r.peak_vram_mb for r in bs1)

        entry = {"config": cid, "avg_p95_ms": avg_p95, "avg_tps": avg_tps, "max_vram_mb": avg_vram}

        if best_latency is None or avg_p95 < best_latency["avg_p95_ms"]:
            best_latency = entry
        if best_throughput is None or avg_tps > best_throughput["avg_tps"]:
            best_throughput = entry
        if best_vram is None or avg_vram < best_vram["max_vram_mb"]:
            best_vram = entry

    if best_latency:
        guidelines["recommendations"].append({
            "use_case": "Lowest latency",
            "config": best_latency["config"],
            "avg_p95_ms": round(best_latency["avg_p95_ms"], 1),
        })
    if best_throughput:
        guidelines["recommendations"].append({
            "use_case": "Highest throughput",
            "config": best_throughput["config"],
            "avg_tokens_per_s": round(best_throughput["avg_tps"], 1),
        })
    if best_vram:
        guidelines["recommendations"].append({
            "use_case": "Lowest VRAM usage",
            "config": best_vram["config"],
            "max_vram_mb": round(best_vram["max_vram_mb"], 1),
        })

    guidelines["general_advice"] = [
        "Always use 4-bit weight quantization on 4 GB VRAM GPUs.",
        "Start with batch_size=1 for reliability; batch_size=4 may OOM at longer contexts.",
        "Monitor GPU temperature; laptop GPUs throttle under sustained load.",
        "Use SDPA attention when available for lower VRAM usage.",
        "KV-cache quantization (INT4) can save VRAM at longer context lengths with minimal quality cost.",
    ]

    save_results_json(guidelines, os.path.join(WEEK_DIR, "configuration_guidelines.json"))
    return guidelines


def main():
    logger.info("=" * 60)
    logger.info("WEEK 10 — Full Benchmark Suite")
    logger.info("=" * 60)

    set_seed(BENCHMARK_CFG.seed)

    # 1. Run full benchmark suite
    all_results, oom_thresholds = run_full_benchmark_suite()

    # 2. Consolidated table
    table_rows = generate_consolidated_table(all_results)

    # 3. OOM thresholds
    save_results_json(oom_thresholds, os.path.join(WEEK_DIR, "oom_thresholds.json"))

    # 4. Configuration guidelines
    guidelines = generate_configuration_guidelines(all_results, oom_thresholds)

    # Summary
    summary = {
        "week": "10",
        "title": "Full Benchmark Suite",
        "status": "COMPLETE",
        "total_benchmark_rows": len(table_rows),
        "configurations_benchmarked": len(set(r.config_id for r in all_results)),
        "oom_thresholds": {
            k: v.get("oom_threshold_tokens") for k, v in oom_thresholds.items()
        },
        "deliverables": [
            "final_benchmark_table.csv",
            "final_benchmark_table.json",
            "oom_thresholds.json",
            "configuration_guidelines.json",
        ],
    }
    save_results_json(summary, os.path.join(WEEK_DIR, "week10_summary.json"))

    logger.info("=" * 60)
    logger.info("Week 10 deliverables saved to: " + WEEK_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
