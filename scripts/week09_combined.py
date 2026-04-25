"""
Week 09 — Combined Configuration

Tests combined FlashAttention-2 + KV-cache quantization.

Deliverables:
  - Combined configuration results if stable
  - Or documented incompatibility report with evidence
  - Final benchmark runbook, fixed configs, and scripts

Usage:
  python scripts/week09_combined.py
  python scripts/week09_combined.py --model qwen2.5-3b
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODEL_CANDIDATES, PRIMARY_BENCHMARK_MODELS, BENCHMARK_CFG, CORRECTNESS_CFG,
    RESULTS_DIR, PROMPTS_DIR,
)
from src.utils import setup_logging, save_results_json, set_seed
from src.model_loader import load_model_4bit
from src.attention_backends import sdpa_backend, check_attention_backends
from src.kv_cache_quant import run_inference_with_kv_quant
from src.benchmark_harness import (
    run_single_inference, run_benchmark_sweep,
    prepare_prompt_at_length, results_to_table,
)
from src.correctness import run_correctness_suite

logger = setup_logging("week09")


def _load_model_for_attn(model_key, attn_implementation):
    model_info = MODEL_CANDIDATES[model_key]
    model_id = model_info["hf_id"]
    if model_info.get("multimodal", False):
        from src.multimodal_loader import load_multimodal_model_4bit
        model, processor = load_multimodal_model_4bit(
            model_key, attn_implementation=attn_implementation
        )
        tokenizer = getattr(processor, "tokenizer", processor)
    else:
        model, tokenizer = load_model_4bit(model_id, attn_implementation=attn_implementation)
    return model, tokenizer


def test_combined_stability(model, tokenizer, model_key, out_dir):
    """Test whether combined attention + KV quant is stable."""
    logger.info("Testing combined configuration stability...")

    with open(os.path.join(PROMPTS_DIR, "fixed_prompts.json")) as f:
        prompts = json.load(f)
    prompt = prompts["single_turn"][0]["text"]

    available = check_attention_backends()

    if available.get("flash_sdp_available"):
        attn_backend = "flash"
    elif available.get("mem_efficient_sdp_available"):
        attn_backend = "mem_efficient"
    else:
        attn_backend = "math"

    combinations = [
        {"attention": attn_backend, "kv_quant": "int4", "id": f"combined_{attn_backend}_i4"},
        {"attention": attn_backend, "kv_quant": "int2", "id": f"combined_{attn_backend}_i2"},
    ]

    stability_results = {}

    for combo in combinations:
        logger.info(
            f"  Testing: attention={combo['attention']}, kv={combo['kv_quant']}"
        )
        try:
            with sdpa_backend(combo["attention"]):
                result = run_inference_with_kv_quant(
                    model=model, tokenizer=tokenizer,
                    prompt=prompt,
                    kv_quant_type=combo["kv_quant"],
                    max_new_tokens=64,
                )

            stability_results[combo["id"]] = {
                "stable": result.get("status") == "success",
                "attention_backend": combo["attention"],
                "kv_quant_type": combo["kv_quant"],
                "test_result": result,
            }
            logger.info(f"    -> Stable: {result.get('status') == 'success'}")

        except Exception as e:
            stability_results[combo["id"]] = {
                "stable": False,
                "attention_backend": combo["attention"],
                "kv_quant_type": combo["kv_quant"],
                "error": str(e),
            }
            logger.info(f"    -> Failed: {e}")

    save_results_json(
        stability_results,
        os.path.join(out_dir, "combined_stability_test.json"),
    )
    return stability_results


def test_fa2_combined_stability(model_key, out_dir):
    """Test FlashAttention-2 + KV-cache combined configs explicitly."""
    import gc
    import torch

    logger.info("Testing FlashAttention-2 + KV-cache combined stability...")

    fa2_results = {}
    try:
        model, tokenizer = _load_model_for_attn(model_key, attn_implementation="flash_attention_2")
    except Exception as e:
        logger.warning(f"FA2 not available for combined test: {e}")
        return fa2_results

    with open(os.path.join(PROMPTS_DIR, "fixed_prompts.json")) as f:
        prompts = json.load(f)
    prompt = prompts["single_turn"][0]["text"]

    for kv_qt in ["int4", "int2"]:
        config_id = f"combined_fa2_{kv_qt[3:]}"
        logger.info(f"  Testing: FA2 + KV {kv_qt}")
        try:
            result = run_inference_with_kv_quant(
                model=model, tokenizer=tokenizer,
                prompt=prompt,
                kv_quant_type=kv_qt,
                max_new_tokens=64,
            )
            fa2_results[config_id] = {
                "stable": result.get("status") == "success",
                "attention_backend": "flash_attention_2",
                "kv_quant_type": kv_qt,
                "test_result": result,
            }
            logger.info(f"    -> Stable: {result.get('status') == 'success'}")
        except Exception as e:
            fa2_results[config_id] = {
                "stable": False,
                "attention_backend": "flash_attention_2",
                "kv_quant_type": kv_qt,
                "error": str(e),
            }
            logger.info(f"    -> Failed: {e}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    save_results_json(fa2_results, os.path.join(out_dir, "fa2_combined_stability.json"))
    return fa2_results


def run_combined_benchmarks(model, tokenizer, stability_results, out_dir):
    logger.info("Running combined configuration benchmarks...")

    with open(os.path.join(PROMPTS_DIR, "fixed_prompts.json")) as f:
        prompts = json.load(f)
    base_prompt = prompts["single_turn"][0]["text"]

    all_results = []

    for config_id, info in stability_results.items():
        if not info.get("stable", False):
            logger.info(f"  Skipping unstable config: {config_id}")
            continue

        logger.info(f"  Benchmarking: {config_id}")

        with sdpa_backend(info["attention_backend"]):
            results = run_benchmark_sweep(
                model=model, tokenizer=tokenizer,
                base_prompt=base_prompt,
                config_id=config_id,
                prompt_lengths=BENCHMARK_CFG.prompt_lengths,
                output_limit=BENCHMARK_CFG.output_limit,
                batch_sizes=[1],
                num_warmup=BENCHMARK_CFG.num_warmup_runs,
                num_runs=BENCHMARK_CFG.num_benchmark_runs,
                seed=BENCHMARK_CFG.seed,
            )
            all_results.extend(results)

    if all_results:
        table_rows = results_to_table(all_results)
        save_results_json(table_rows, os.path.join(out_dir, "combined_benchmark_results.json"))

    return all_results


def generate_incompatibility_report(stability_results, out_dir):
    failures = {
        k: v for k, v in stability_results.items() if not v.get("stable", False)
    }

    if not failures:
        report = {
            "status": "ALL_COMPATIBLE",
            "message": "All tested combinations are stable.",
        }
    else:
        report = {
            "status": "SOME_INCOMPATIBLE",
            "failures": {
                k: {
                    "attention": v.get("attention_backend"),
                    "kv_quant": v.get("kv_quant_type"),
                    "error": v.get("error", v.get("test_result", {}).get("error", "unknown")),
                }
                for k, v in failures.items()
            },
            "recommendation": (
                "Report separate gains from attention optimization and KV-cache "
                "quantization independently."
            ),
        }

    save_results_json(report, os.path.join(out_dir, "incompatibility_report.json"))
    return report


def generate_final_benchmark_runbook(stability_results, out_dir):
    logger.info("Generating final benchmark runbook...")

    configs = [
        {
            "id": "baseline",
            "description": "FP16 KV-cache, eager attention",
            "attn_implementation": "eager",
            "kv_quant": "fp16",
        },
    ]

    available = check_attention_backends()
    configs.append({
        "id": "sdpa_default",
        "description": "FP16 KV-cache, SDPA (auto backend)",
        "attn_implementation": "sdpa",
        "kv_quant": "fp16",
    })

    if available.get("flash_sdp_available"):
        configs.append({
            "id": "sdpa_flash",
            "description": "FP16 KV-cache, SDPA flash attention",
            "attn_implementation": "sdpa",
            "sdpa_backend": "flash",
            "kv_quant": "fp16",
        })

    # FA2 as first-class config
    configs.append({
        "id": "flash_attention_2",
        "description": "FP16 KV-cache, FlashAttention-2",
        "attn_implementation": "flash_attention_2",
        "kv_quant": "fp16",
    })

    configs.append({
        "id": "kv_int4",
        "description": "INT4 KV-cache, eager attention",
        "attn_implementation": "eager",
        "kv_quant": "int4",
    })

    for config_id, info in stability_results.items():
        if info.get("stable", False):
            configs.append({
                "id": config_id,
                "description": f"{info['kv_quant_type'].upper()} KV-cache, "
                               f"{info['attention_backend']} attention",
                "attn_implementation": (
                    "flash_attention_2" if info["attention_backend"] == "flash_attention_2"
                    else "sdpa"
                ),
                "sdpa_backend": info["attention_backend"] if info["attention_backend"] != "flash_attention_2" else None,
                "kv_quant": info["kv_quant_type"],
            })

    runbook = {
        "configurations": configs,
        "benchmark_settings": {
            "prompt_lengths": BENCHMARK_CFG.prompt_lengths,
            "output_limit": BENCHMARK_CFG.output_limit,
            "batch_sizes": BENCHMARK_CFG.batch_sizes,
            "num_warmup": BENCHMARK_CFG.num_warmup_runs,
            "num_runs": BENCHMARK_CFG.num_benchmark_runs,
            "seed": BENCHMARK_CFG.seed,
        },
        "metrics": [
            "p50_latency_ms", "p95_latency_ms", "throughput_tok_per_s",
            "peak_vram_mb", "peak_cpu_ram_mb", "oom_threshold_tokens",
        ],
    }

    save_results_json(runbook, os.path.join(out_dir, "final_benchmark_runbook.json"))
    logger.info(f"  {len(configs)} configurations in final runbook")
    return runbook


def run_for_model(model_key: str):
    import torch
    import gc

    model_info = MODEL_CANDIDATES[model_key]
    model_id = model_info["hf_id"]

    out_dir = os.path.join(RESULTS_DIR, "week09_combined", model_key)
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Loading model with SDPA: {model_id}")
    model, tokenizer = _load_model_for_attn(model_key, attn_implementation="sdpa")

    stability_results = test_combined_stability(model, tokenizer, model_key, out_dir)
    combined_results = run_combined_benchmarks(model, tokenizer, stability_results, out_dir)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Test FA2 + KV-cache combined (requires separate model load)
    fa2_stability = test_fa2_combined_stability(model_key, out_dir)
    stability_results.update(fa2_stability)

    incompat_report = generate_incompatibility_report(stability_results, out_dir)
    runbook = generate_final_benchmark_runbook(stability_results, out_dir)

    stable_count = sum(1 for v in stability_results.values() if v.get("stable"))
    summary = {
        "week": "09",
        "title": "Combined Configuration",
        "status": "COMPLETE",
        "model_key": model_key,
        "combinations_tested": len(stability_results),
        "combinations_stable": stable_count,
        "fa2_combined_tested": bool(fa2_stability),
        "incompatibility_status": incompat_report["status"],
        "final_configs_count": len(runbook["configurations"]),
        "deliverables": [
            "combined_stability_test.json",
            "fa2_combined_stability.json",
            "combined_benchmark_results.json",
            "incompatibility_report.json",
            "final_benchmark_runbook.json",
        ],
    }
    save_results_json(summary, os.path.join(out_dir, "week09_summary.json"))


def main(model_keys=None):
    from src.utils import CheckpointManager
    ckpt = CheckpointManager("week09")

    logger.info("=" * 60)
    logger.info("WEEK 09 — Combined Configuration")
    logger.info("=" * 60)

    set_seed(BENCHMARK_CFG.seed)

    if model_keys is None:
        model_keys = list(PRIMARY_BENCHMARK_MODELS)

    for model_key in model_keys:
        if ckpt.is_done(model_key):
            logger.info(f"[SKIP] {model_key} already done (checkpoint)")
            continue
        logger.info(f"\n{'#' * 50}")
        logger.info(f"# Model: {model_key}")
        logger.info(f"{'#' * 50}")
        try:
            run_for_model(model_key)
            ckpt.mark_done(model_key)
            logger.info(f"[CHECKPOINT] {model_key} saved")
        except Exception as e:
            logger.error(f"Failed for {model_key}: {e}")
            import traceback
            traceback.print_exc()

    logger.info("=" * 60)
    logger.info("Week 09 complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs="+", default=None)
    args = parser.parse_args()
    main(model_keys=args.model)
