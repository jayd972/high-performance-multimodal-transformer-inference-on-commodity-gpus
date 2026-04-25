"""
Week 07 — KV-Cache Quantization Phase

Deliverables:
  - KV-cache quantization prototype running end to end
  - Correctness metrics: agreement rate and logit difference
  - Memory per token measurement
  - Decision on whether deeper integration is pursued

Usage:
  python scripts/week07_kv_cache.py
  python scripts/week07_kv_cache.py --model qwen2.5-3b
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODEL_CANDIDATES, PRIMARY_BENCHMARK_MODELS, BENCHMARK_CFG, CORRECTNESS_CFG,
    KV_CACHE_QUANT_TYPES, RESULTS_DIR, PROMPTS_DIR,
)
from src.utils import setup_logging, save_results_json, set_seed
from src.model_loader import load_model_4bit
from src.kv_cache_quant import (
    check_kv_quant_support, run_inference_with_kv_quant,
    measure_kv_memory_per_token, compare_kv_quant_configs,
)
from src.correctness import run_correctness_suite

logger = setup_logging("week07")


def check_kv_support(out_dir):
    logger.info("Checking KV-cache quantization support...")
    support = check_kv_quant_support()
    save_results_json(support, os.path.join(out_dir, "kv_quant_support.json"))
    logger.info(f"  Transformers KV quant: {support['transformers_kv_quant']}")
    logger.info(f"  Quanto available: {support['quanto_available']}")
    return support


def test_kv_quant_prototype(model, tokenizer, out_dir):
    logger.info("Testing KV-cache quantization prototype...")

    with open(os.path.join(PROMPTS_DIR, "fixed_prompts.json")) as f:
        prompts = json.load(f)
    prompt = prompts["single_turn"][0]["text"]

    prototype_results = {}
    for qt in ["int4", "int2"]:
        logger.info(f"  Testing KV quant type: {qt}")
        result = run_inference_with_kv_quant(
            model=model, tokenizer=tokenizer,
            prompt=prompt, kv_quant_type=qt,
            max_new_tokens=64,
        )
        prototype_results[qt] = result
        logger.info(f"    Status: {result['status']}")
        if result["status"] == "success":
            logger.info(f"    Latency: {result['total_time_ms']:.0f}ms")
            logger.info(f"    Peak VRAM: {result['peak_vram_mb']:.0f}MB")

    save_results_json(prototype_results, os.path.join(out_dir, "kv_quant_prototype.json"))
    return prototype_results


def run_kv_correctness(model, tokenizer, kv_quant_type, model_key, out_dir):
    logger.info(f"Running correctness check for KV {kv_quant_type}...")

    baseline_path = os.path.join(
        RESULTS_DIR, "week03_baseline", model_key, "baseline_correctness_full.json"
    )
    if not os.path.exists(baseline_path):
        baseline_path = os.path.join(
            RESULTS_DIR, "week03_baseline", "baseline_correctness_full.json"
        )
    baseline_traces = None
    if os.path.exists(baseline_path):
        from src.utils import load_results_json
        data = load_results_json(baseline_path)
        baseline_traces = data.get("traces", [])

    with open(os.path.join(PROMPTS_DIR, "fixed_prompts.json")) as f:
        prompts_data = json.load(f)

    correctness_prompts = [
        p["text"] for p in prompts_data["single_turn"][:CORRECTNESS_CFG.num_fixed_prompts]
    ]

    result = run_correctness_suite(
        model=model, tokenizer=tokenizer,
        prompts=correctness_prompts,
        baseline_traces=baseline_traces,
        num_steps=CORRECTNESS_CFG.num_logit_comparison_steps,
        config_id=f"kv_{kv_quant_type}",
        seed=BENCHMARK_CFG.seed,
    )

    return result


def run_memory_per_token(model, tokenizer, out_dir):
    logger.info("Measuring KV-cache memory per token...")

    with open(os.path.join(PROMPTS_DIR, "fixed_prompts.json")) as f:
        prompts = json.load(f)
    base_prompt = prompts["long_context_seed"]["text"]

    results = {}
    for qt in KV_CACHE_QUANT_TYPES:
        logger.info(f"  Measuring memory per token for KV type: {qt}")
        result = measure_kv_memory_per_token(
            model=model, tokenizer=tokenizer,
            base_prompt=base_prompt,
            kv_quant_type=qt,
            lengths=[64, 128, 256, 512, 768, 1024],
        )
        results[qt] = result
        logger.info(f"    Estimated MB/token: {result['estimated_mb_per_token']}")

    save_results_json(results, os.path.join(out_dir, "kv_memory_per_token.json"))
    return results


def make_deeper_integration_decision(prototype_results, memory_results, out_dir):
    decision = {
        "recommendation": "",
        "rationale": "",
        "available_methods": [],
        "issues_encountered": [],
    }

    successful = []
    for qt, res in prototype_results.items():
        if res.get("status") == "success":
            successful.append(qt)
        else:
            decision["issues_encountered"].append(
                f"{qt}: {res.get('error', 'unknown error')}"
            )

    if "int4" in successful:
        decision["available_methods"].append("int4")
    if "int2" in successful:
        decision["available_methods"].append("int2")

    if successful:
        decision["recommendation"] = "USE_EXISTING"
        decision["rationale"] = (
            f"KV-cache quantization ({', '.join(successful)}) works with the "
            "existing transformers/quanto pipeline. No deeper framework "
            "modification is needed."
        )
    else:
        decision["recommendation"] = "SKIP_OR_USE_LLAMA_CPP"
        decision["rationale"] = (
            "KV-cache quantization via transformers/quanto is not functional. "
            "Consider using llama.cpp as a fallback."
        )

    save_results_json(decision, os.path.join(out_dir, "integration_decision.json"))
    logger.info(f"  Decision: {decision['recommendation']}")
    return decision


def run_for_model(model_key: str):
    import torch

    model_info = MODEL_CANDIDATES[model_key]
    model_id = model_info["hf_id"]
    is_multimodal = model_info.get("multimodal", False)

    out_dir = os.path.join(RESULTS_DIR, "week07_kv_cache", model_key)
    os.makedirs(out_dir, exist_ok=True)

    support = check_kv_support(out_dir)

    logger.info(f"Loading model: {model_id}")
    if is_multimodal:
        from src.multimodal_loader import load_multimodal_model_4bit
        model, processor = load_multimodal_model_4bit(model_key)
        tokenizer = getattr(processor, "tokenizer", processor)
    else:
        model, tokenizer = load_model_4bit(model_id)

    prototype_results = test_kv_quant_prototype(model, tokenizer, out_dir)

    correctness_results = {}
    for qt in ["int4", "int2"]:
        if prototype_results.get(qt, {}).get("status") == "success":
            corr = run_kv_correctness(model, tokenizer, qt, model_key, out_dir)
            correctness_results[qt] = {
                "config_id": corr.get("config_id"),
                "aggregate": corr.get("aggregate", {}),
            }

    save_results_json(correctness_results, os.path.join(out_dir, "kv_correctness_report.json"))

    memory_results = run_memory_per_token(model, tokenizer, out_dir)
    decision = make_deeper_integration_decision(prototype_results, memory_results, out_dir)

    summary = {
        "week": "07",
        "title": "KV-Cache Quantization Phase",
        "status": "COMPLETE",
        "model_key": model_key,
        "kv_quant_support": support,
        "prototype_status": {
            qt: res.get("status") for qt, res in prototype_results.items()
        },
        "integration_decision": decision["recommendation"],
        "deliverables": [
            "kv_quant_support.json",
            "kv_quant_prototype.json",
            "kv_correctness_report.json",
            "kv_memory_per_token.json",
            "integration_decision.json",
        ],
    }
    save_results_json(summary, os.path.join(out_dir, "week07_summary.json"))

    del model
    if is_multimodal:
        del processor
    else:
        del tokenizer
    torch.cuda.empty_cache()


def main(model_keys=None):
    from src.utils import CheckpointManager
    ckpt = CheckpointManager("week07")

    logger.info("=" * 60)
    logger.info("WEEK 07 — KV-Cache Quantization Phase")
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
    logger.info("Week 07 complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs="+", default=None)
    args = parser.parse_args()
    main(model_keys=args.model)
