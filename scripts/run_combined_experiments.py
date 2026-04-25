"""
Combined attention + KV-cache experiments on MHA models.

Runs the full benchmark matrix (baseline, SDPA, KV INT4/INT2, combined)
on models with standard multi-head attention where FlashAttention-2 and
QuantizedCache are expected to compose without kernel conflicts.

Models:
  - microsoft/phi-2            (2.7B, MHA 32Q/32KV)
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0  (1.1B, MHA 32Q/32KV)

Usage:
    python scripts/run_combined_experiments.py                  # both models
    python scripts/run_combined_experiments.py --model phi-2    # single model
    python scripts/run_combined_experiments.py --model tinyllama-1.1b
"""

import os
import sys
import gc
import json
import argparse
import time
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from config import (
    MODEL_CANDIDATES, COMBINED_EXPERIMENT_MODELS,
    BENCHMARK_CFG, CORRECTNESS_CFG, QUALITY_CFG,
    RESULTS_DIR, PROMPTS_DIR, KV_CACHE_QUANT_TYPES,
)
from src.utils import setup_logging, save_results_json, save_results_csv, set_seed
from src.model_loader import load_model_4bit, get_model_memory_footprint, verify_vram_budget
from src.attention_backends import sdpa_backend, check_attention_backends
from src.benchmark_harness import (
    run_benchmark_sweep, find_oom_threshold,
    run_single_inference, prepare_prompt_at_length,
    results_to_table, BenchmarkResult,
)
from src.kv_cache_quant import (
    check_kv_quant_support, run_inference_with_kv_quant,
    compare_kv_quant_configs, measure_kv_memory_per_token,
)
from src.correctness import run_correctness_suite
from src.quality_eval import run_quality_evaluation

logger = setup_logging("combined_experiments")


def _unload(model, tokenizer=None):
    """Free GPU memory."""
    del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_prompts():
    with open(os.path.join(PROMPTS_DIR, "fixed_prompts.json")) as f:
        return json.load(f)


def run_model_experiments(model_key: str):
    """Run the full experiment suite on a single model."""
    info = MODEL_CANDIDATES[model_key]
    model_id = info["hf_id"]

    out_dir = os.path.join(RESULTS_DIR, f"combined_{model_key.replace('.', '_')}")
    os.makedirs(out_dir, exist_ok=True)

    prompts_data = load_prompts()
    base_prompt = prompts_data["single_turn"][0]["text"]
    long_prompt = prompts_data["long_context_seed"]["text"]
    correctness_prompts = [
        p["text"] for p in prompts_data["single_turn"][:CORRECTNESS_CFG.num_fixed_prompts]
    ]

    logger.info("=" * 70)
    logger.info(f"MODEL: {model_id}  ({info['params_b']}B, {info['attention']})")
    logger.info("=" * 70)

    all_benchmark_results = []
    oom_thresholds = {}
    correctness_results = {}
    quality_results = {}
    stability_results = {}

    # ── 0. Environment / availability ────────────────────────────────
    env = check_attention_backends()
    kv_support = check_kv_quant_support()
    save_results_json(
        {"attention": env, "kv_cache": kv_support, "model": info},
        os.path.join(out_dir, "environment.json"),
    )

    # ── 1. Baseline (eager, FP16 KV) ────────────────────────────────
    logger.info("\n--- [1/7] Baseline (eager, FP16 KV) ---")
    model, tokenizer = load_model_4bit(model_id, attn_implementation="eager")
    footprint = verify_vram_budget(model)
    save_results_json(footprint, os.path.join(out_dir, "model_footprint.json"))

    results = run_benchmark_sweep(
        model, tokenizer, base_prompt,
        config_id="baseline",
        prompt_lengths=BENCHMARK_CFG.prompt_lengths,
        output_limit=BENCHMARK_CFG.output_limit,
        batch_sizes=BENCHMARK_CFG.batch_sizes,
        num_warmup=BENCHMARK_CFG.num_warmup_runs,
        num_runs=BENCHMARK_CFG.num_benchmark_runs,
        seed=BENCHMARK_CFG.seed,
    )
    all_benchmark_results.extend(results)

    oom = find_oom_threshold(
        model, tokenizer, long_prompt,
        max_new_tokens=BENCHMARK_CFG.output_limit,
        start_length=128, max_length=8192, step=256,
    )
    oom_thresholds["baseline"] = oom

    baseline_correctness = run_correctness_suite(
        model, tokenizer, correctness_prompts,
        baseline_traces=None,
        num_steps=CORRECTNESS_CFG.num_logit_comparison_steps,
        config_id="baseline", seed=BENCHMARK_CFG.seed,
    )
    baseline_traces = baseline_correctness.get("traces", [])

    baseline_quality = run_quality_evaluation(
        model, tokenizer,
        dataset_names=list(QUALITY_CFG.datasets),
        num_examples=QUALITY_CFG.num_examples_per_dataset,
        config_id="baseline", seed=BENCHMARK_CFG.seed,
    )
    quality_results["baseline"] = baseline_quality
    _unload(model, tokenizer)

    # ── 2. SDPA default (FP16 KV) ───────────────────────────────────
    logger.info("\n--- [2/7] SDPA default (FP16 KV) ---")
    model, tokenizer = load_model_4bit(model_id, attn_implementation="sdpa")

    results = run_benchmark_sweep(
        model, tokenizer, base_prompt,
        config_id="sdpa_default",
        prompt_lengths=BENCHMARK_CFG.prompt_lengths,
        output_limit=BENCHMARK_CFG.output_limit,
        batch_sizes=BENCHMARK_CFG.batch_sizes,
        num_warmup=BENCHMARK_CFG.num_warmup_runs,
        num_runs=BENCHMARK_CFG.num_benchmark_runs,
        seed=BENCHMARK_CFG.seed,
    )
    all_benchmark_results.extend(results)

    oom = find_oom_threshold(
        model, tokenizer, long_prompt,
        max_new_tokens=BENCHMARK_CFG.output_limit,
        start_length=128, max_length=8192, step=256,
    )
    oom_thresholds["sdpa_default"] = oom
    _unload(model, tokenizer)

    # ── 3. KV INT4 (eager) ──────────────────────────────────────────
    logger.info("\n--- [3/7] KV INT4 (eager attention) ---")
    model, tokenizer = load_model_4bit(model_id, attn_implementation="eager")

    for prompt_len in BENCHMARK_CFG.prompt_lengths:
        prompt = prepare_prompt_at_length(tokenizer, base_prompt, prompt_len)
        kv_result = compare_kv_quant_configs(
            model, tokenizer, prompt,
            quant_types=["int4"],
            max_new_tokens=BENCHMARK_CFG.output_limit,
            num_runs=BENCHMARK_CFG.num_benchmark_runs,
        )
        qt_data = kv_result.get("int4", {})
        bench = BenchmarkResult(
            config_id="kv_int4",
            prompt_length=prompt_len,
            output_limit=BENCHMARK_CFG.output_limit,
            batch_size=1,
            num_runs=qt_data.get("num_runs", 0),
            latency_p50_ms=qt_data.get("latency_p50_ms", 0),
            latency_p95_ms=qt_data.get("latency_p95_ms", 0),
            latency_mean_ms=qt_data.get("latency_mean_ms", 0),
            throughput_tokens_per_s=round(
                BENCHMARK_CFG.output_limit / (qt_data.get("latency_mean_ms", 1) / 1000), 2
            ) if qt_data.get("latency_mean_ms", 0) > 0 else 0,
            peak_vram_mb=qt_data.get("peak_vram_mb", 0),
            oom_occurred=qt_data.get("status") != "success",
        )
        all_benchmark_results.append(bench)

    kv_mem = measure_kv_memory_per_token(
        model, tokenizer, long_prompt,
        kv_quant_type="int4",
        lengths=[64, 128, 256, 512, 768, 1024],
    )
    save_results_json(kv_mem, os.path.join(out_dir, "kv_int4_memory_per_token.json"))
    _unload(model, tokenizer)

    # ── 4. Combined: SDPA + KV INT4 ─────────────────────────────────
    logger.info("\n--- [4/7] Combined: SDPA + KV INT4 ---")
    model, tokenizer = load_model_4bit(model_id, attn_implementation="sdpa")

    for kv_qt in ["int4", "int2"]:
        config_id = f"combined_sdpa_{kv_qt[-2:]}"
        logger.info(f"  Testing combined: SDPA + KV {kv_qt}")

        test_result = run_inference_with_kv_quant(
            model, tokenizer, base_prompt,
            kv_quant_type=kv_qt, max_new_tokens=32,
        )
        stable = test_result.get("status") == "success"
        stability_results[config_id] = {
            "stable": stable,
            "attention": "sdpa",
            "kv_quant": kv_qt,
            "test_result": test_result,
        }
        logger.info(f"    Stable: {stable}")

        if stable:
            for prompt_len in BENCHMARK_CFG.prompt_lengths:
                prompt = prepare_prompt_at_length(tokenizer, base_prompt, prompt_len)
                kv_result = compare_kv_quant_configs(
                    model, tokenizer, prompt,
                    quant_types=[kv_qt],
                    max_new_tokens=BENCHMARK_CFG.output_limit,
                    num_runs=BENCHMARK_CFG.num_benchmark_runs,
                )
                qt_data = kv_result.get(kv_qt, {})
                bench = BenchmarkResult(
                    config_id=config_id,
                    prompt_length=prompt_len,
                    output_limit=BENCHMARK_CFG.output_limit,
                    batch_size=1,
                    num_runs=qt_data.get("num_runs", 0),
                    latency_p50_ms=qt_data.get("latency_p50_ms", 0),
                    latency_p95_ms=qt_data.get("latency_p95_ms", 0),
                    latency_mean_ms=qt_data.get("latency_mean_ms", 0),
                    throughput_tokens_per_s=round(
                        BENCHMARK_CFG.output_limit / (qt_data.get("latency_mean_ms", 1) / 1000), 2
                    ) if qt_data.get("latency_mean_ms", 0) > 0 else 0,
                    peak_vram_mb=qt_data.get("peak_vram_mb", 0),
                    oom_occurred=qt_data.get("status") != "success",
                )
                all_benchmark_results.append(bench)

            oom = find_oom_threshold(
                model, tokenizer, long_prompt,
                max_new_tokens=BENCHMARK_CFG.output_limit,
                start_length=128, max_length=8192, step=256,
            )
            oom_thresholds[config_id] = oom

    _unload(model, tokenizer)

    # ── 5. Combined: flash_attention_2 + KV INT4/INT2 ───────────────
    logger.info("\n--- [5/7] Combined: flash_attention_2 + KV INT4/INT2 ---")
    try:
        model, tokenizer = load_model_4bit(model_id, attn_implementation="flash_attention_2")
        flash_available = True
    except Exception as e:
        logger.warning(f"flash_attention_2 not available for {model_id}: {e}")
        flash_available = False

    if flash_available:
        results = run_benchmark_sweep(
            model, tokenizer, base_prompt,
            config_id="flash_attention_2",
            prompt_lengths=BENCHMARK_CFG.prompt_lengths,
            output_limit=BENCHMARK_CFG.output_limit,
            batch_sizes=BENCHMARK_CFG.batch_sizes,
            num_warmup=BENCHMARK_CFG.num_warmup_runs,
            num_runs=BENCHMARK_CFG.num_benchmark_runs,
            seed=BENCHMARK_CFG.seed,
        )
        all_benchmark_results.extend(results)

        oom = find_oom_threshold(
            model, tokenizer, long_prompt,
            max_new_tokens=BENCHMARK_CFG.output_limit,
            start_length=128, max_length=8192, step=256,
        )
        oom_thresholds["flash_attention_2"] = oom

        for kv_qt in ["int4", "int2"]:
            config_id = f"combined_fa2_{kv_qt[-2:]}"
            logger.info(f"  Testing combined: flash_attention_2 + KV {kv_qt}")

            test_result = run_inference_with_kv_quant(
                model, tokenizer, base_prompt,
                kv_quant_type=kv_qt, max_new_tokens=32,
            )
            stable = test_result.get("status") == "success"
            stability_results[config_id] = {
                "stable": stable,
                "attention": "flash_attention_2",
                "kv_quant": kv_qt,
                "test_result": test_result,
            }
            logger.info(f"    Stable: {stable}")

            if stable:
                for prompt_len in BENCHMARK_CFG.prompt_lengths:
                    prompt = prepare_prompt_at_length(tokenizer, base_prompt, prompt_len)
                    kv_result = compare_kv_quant_configs(
                        model, tokenizer, prompt,
                        quant_types=[kv_qt],
                        max_new_tokens=BENCHMARK_CFG.output_limit,
                        num_runs=BENCHMARK_CFG.num_benchmark_runs,
                    )
                    qt_data = kv_result.get(kv_qt, {})
                    bench = BenchmarkResult(
                        config_id=config_id,
                        prompt_length=prompt_len,
                        output_limit=BENCHMARK_CFG.output_limit,
                        batch_size=1,
                        num_runs=qt_data.get("num_runs", 0),
                        latency_p50_ms=qt_data.get("latency_p50_ms", 0),
                        latency_p95_ms=qt_data.get("latency_p95_ms", 0),
                        latency_mean_ms=qt_data.get("latency_mean_ms", 0),
                        throughput_tokens_per_s=round(
                            BENCHMARK_CFG.output_limit / (qt_data.get("latency_mean_ms", 1) / 1000), 2
                        ) if qt_data.get("latency_mean_ms", 0) > 0 else 0,
                        peak_vram_mb=qt_data.get("peak_vram_mb", 0),
                        oom_occurred=qt_data.get("status") != "success",
                    )
                    all_benchmark_results.append(bench)

                oom = find_oom_threshold(
                    model, tokenizer, long_prompt,
                    max_new_tokens=BENCHMARK_CFG.output_limit,
                    start_length=128, max_length=8192, step=256,
                )
                oom_thresholds[config_id] = oom

        _unload(model, tokenizer)

    # ── 6. Quality evaluation on stable combined configs ─────────────
    logger.info("\n--- [6/7] Quality evaluation on stable combined configs ---")
    stable_combined = [k for k, v in stability_results.items() if v.get("stable")]
    for config_id in stable_combined:
        info_c = stability_results[config_id]
        attn = info_c["attention"]
        kv_qt = info_c["kv_quant"]

        attn_impl = "flash_attention_2" if "flash" in attn else "sdpa"
        try:
            model, tokenizer = load_model_4bit(model_id, attn_implementation=attn_impl)
        except Exception as e:
            logger.warning(f"Could not load model for {config_id}: {e}")
            continue

        q_result = run_quality_evaluation(
            model, tokenizer,
            dataset_names=list(QUALITY_CFG.datasets),
            num_examples=QUALITY_CFG.num_examples_per_dataset,
            config_id=config_id, seed=BENCHMARK_CFG.seed,
        )
        quality_results[config_id] = q_result
        _unload(model, tokenizer)

    # ── 7. Save all results ──────────────────────────────────────────
    logger.info("\n--- [7/7] Saving results ---")

    table_rows = results_to_table(all_benchmark_results)
    save_results_csv(table_rows, os.path.join(out_dir, "benchmark_table.csv"))
    save_results_json(table_rows, os.path.join(out_dir, "benchmark_table.json"))
    save_results_json(oom_thresholds, os.path.join(out_dir, "oom_thresholds.json"))
    save_results_json(stability_results, os.path.join(out_dir, "stability_results.json"))
    save_results_json(quality_results, os.path.join(out_dir, "quality_results.json"))

    stable_count = sum(1 for v in stability_results.values() if v.get("stable"))
    summary = {
        "model_key": model_key,
        "model_id": model_id,
        "attention_type": info["attention"],
        "timestamp": datetime.datetime.now().isoformat(),
        "configurations_tested": len(set(r.config_id for r in all_benchmark_results)),
        "total_benchmark_rows": len(table_rows),
        "combined_stability": {k: v.get("stable") for k, v in stability_results.items()},
        "combined_stable_count": stable_count,
        "oom_thresholds": {k: v.get("oom_threshold_tokens") for k, v in oom_thresholds.items()},
        "quality_configs_evaluated": list(quality_results.keys()),
        "deliverables": [
            "benchmark_table.csv", "benchmark_table.json",
            "oom_thresholds.json", "stability_results.json",
            "quality_results.json", "environment.json",
            "model_footprint.json", "kv_int4_memory_per_token.json",
        ],
    }
    save_results_json(summary, os.path.join(out_dir, "summary.json"))

    logger.info(f"\nResults saved to: {out_dir}")
    logger.info(f"Configs tested: {summary['configurations_tested']}")
    logger.info(f"Combined stable: {stable_count}/{len(stability_results)}")
    logger.info(f"OOM thresholds: {summary['oom_thresholds']}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run combined attention + KV-cache experiments on MHA models",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=COMBINED_EXPERIMENT_MODELS,
        help="Run on a single model (default: all)",
    )
    args = parser.parse_args()

    set_seed(BENCHMARK_CFG.seed)

    models_to_run = [args.model] if args.model else list(COMBINED_EXPERIMENT_MODELS)

    logger.info("=" * 70)
    logger.info("COMBINED ATTENTION + KV-CACHE EXPERIMENTS")
    logger.info(f"Models: {models_to_run}")
    logger.info("=" * 70)

    all_summaries = {}
    for model_key in models_to_run:
        try:
            summary = run_model_experiments(model_key)
            all_summaries[model_key] = summary
        except Exception as e:
            logger.error(f"FAILED for {model_key}: {e}")
            import traceback
            traceback.print_exc()
            all_summaries[model_key] = {"error": str(e)}

    save_results_json(
        all_summaries,
        os.path.join(RESULTS_DIR, "combined_experiments_summary.json"),
    )

    logger.info("\n" + "=" * 70)
    logger.info("ALL EXPERIMENTS COMPLETE")
    for mk, s in all_summaries.items():
        if "error" in s:
            logger.info(f"  {mk}: FAILED — {s['error']}")
        else:
            logger.info(
                f"  {mk}: {s['combined_stable_count']} combined configs stable, "
                f"OOM thresholds: {s['oom_thresholds']}"
            )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
