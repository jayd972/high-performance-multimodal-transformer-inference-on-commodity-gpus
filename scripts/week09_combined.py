"""
Week 09 — Combined Configuration

Deliverables:
  - Combined configuration results if stable
  - Or documented incompatibility report with evidence
  - Final benchmark runbook, fixed configs, and scripts
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DEFAULT_MODEL_ID, BENCHMARK_CFG, CORRECTNESS_CFG,
    CONFIGURATION_IDS, RESULTS_DIR, PROMPTS_DIR,
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

WEEK_DIR = os.path.join(RESULTS_DIR, "week09_combined")
os.makedirs(WEEK_DIR, exist_ok=True)


def test_combined_stability(model, tokenizer):
    """Test whether combined attention + KV quant is stable."""
    logger.info("Testing combined configuration stability...")

    with open(os.path.join(PROMPTS_DIR, "fixed_prompts.json")) as f:
        prompts = json.load(f)
    prompt = prompts["single_turn"][0]["text"]

    available = check_attention_backends()

    # Determine best attention backend
    if available.get("flash_sdp_available"):
        attn_backend = "flash"
    elif available.get("mem_efficient_sdp_available"):
        attn_backend = "mem_efficient"
    else:
        attn_backend = "math"

    # Test combinations
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
            logger.info(f"    → Stable: {result.get('status') == 'success'}")

        except Exception as e:
            stability_results[combo["id"]] = {
                "stable": False,
                "attention_backend": combo["attention"],
                "kv_quant_type": combo["kv_quant"],
                "error": str(e),
            }
            logger.info(f"    → Failed: {e}")

    save_results_json(
        stability_results,
        os.path.join(WEEK_DIR, "combined_stability_test.json"),
    )

    return stability_results


def run_combined_benchmarks(model, tokenizer, stability_results):
    """Run full benchmarks for stable combined configurations."""
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
                batch_sizes=[1],  # Batch=1 for combined configs
                num_warmup=BENCHMARK_CFG.num_warmup_runs,
                num_runs=BENCHMARK_CFG.num_benchmark_runs,
                seed=BENCHMARK_CFG.seed,
            )
            all_results.extend(results)

    if all_results:
        table_rows = results_to_table(all_results)
        save_results_json(table_rows, os.path.join(WEEK_DIR, "combined_benchmark_results.json"))

    return all_results


def generate_incompatibility_report(stability_results):
    """Document any incompatibilities found."""
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
                "quantization independently, with a clear technical explanation "
                "of why the combination is unstable."
            ),
        }

    save_results_json(report, os.path.join(WEEK_DIR, "incompatibility_report.json"))
    return report


def generate_final_benchmark_runbook(stability_results):
    """Define final benchmark matrix and freeze configs."""
    logger.info("Generating final benchmark runbook...")

    configs = [
        {
            "id": "baseline",
            "description": "FP16 KV-cache, eager attention",
            "attn_implementation": "eager",
            "kv_quant": "fp16",
        },
    ]

    # Add attention configs
    available = check_attention_backends()
    if available.get("flash_sdp_available"):
        configs.append({
            "id": "sdpa_flash",
            "description": "FP16 KV-cache, SDPA flash attention",
            "attn_implementation": "sdpa",
            "sdpa_backend": "flash",
            "kv_quant": "fp16",
        })
    if available.get("mem_efficient_sdp_available"):
        configs.append({
            "id": "sdpa_mem_eff",
            "description": "FP16 KV-cache, SDPA memory-efficient attention",
            "attn_implementation": "sdpa",
            "sdpa_backend": "mem_efficient",
            "kv_quant": "fp16",
        })

    # Add KV quant configs
    configs.append({
        "id": "kv_int4",
        "description": "INT4 KV-cache, eager attention",
        "attn_implementation": "eager",
        "kv_quant": "int4",
    })

    # Add stable combined configs
    for config_id, info in stability_results.items():
        if info.get("stable", False):
            configs.append({
                "id": config_id,
                "description": f"{info['kv_quant_type'].upper()} KV-cache, "
                               f"SDPA {info['attention_backend']} attention",
                "attn_implementation": "sdpa",
                "sdpa_backend": info["attention_backend"],
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

    save_results_json(runbook, os.path.join(WEEK_DIR, "final_benchmark_runbook.json"))
    logger.info(f"  {len(configs)} configurations in final runbook")
    return runbook


def main():
    logger.info("=" * 60)
    logger.info("WEEK 09 — Combined Configuration")
    logger.info("=" * 60)

    set_seed(BENCHMARK_CFG.seed)

    logger.info(f"Loading model with SDPA: {DEFAULT_MODEL_ID}")
    model, tokenizer = load_model_4bit(DEFAULT_MODEL_ID, attn_implementation="sdpa")

    # 1. Stability test
    stability_results = test_combined_stability(model, tokenizer)

    # 2. Combined benchmarks (for stable configs)
    combined_results = run_combined_benchmarks(model, tokenizer, stability_results)

    # 3. Incompatibility report
    incompat_report = generate_incompatibility_report(stability_results)

    # 4. Final benchmark runbook
    runbook = generate_final_benchmark_runbook(stability_results)

    # Summary
    stable_count = sum(1 for v in stability_results.values() if v.get("stable"))
    summary = {
        "week": "09",
        "title": "Combined Configuration",
        "status": "COMPLETE",
        "combinations_tested": len(stability_results),
        "combinations_stable": stable_count,
        "incompatibility_status": incompat_report["status"],
        "final_configs_count": len(runbook["configurations"]),
        "deliverables": [
            "combined_stability_test.json",
            "combined_benchmark_results.json",
            "incompatibility_report.json",
            "final_benchmark_runbook.json",
        ],
    }
    save_results_json(summary, os.path.join(WEEK_DIR, "week09_summary.json"))

    logger.info("=" * 60)
    logger.info("Week 09 deliverables saved to: " + WEEK_DIR)
    logger.info("=" * 60)

    import torch
    del model, tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
