"""
Week Multimodal — Multimodal Model Benchmarks

Runs the full benchmark pipeline for vision-language models with image+text
inputs, including:
  - Baseline (eager attention)
  - SDPA attention
  - FlashAttention-2 (explicit, not SDPA fallback)
  - KV-cache quantization (INT4, INT2)
  - Combined FA2 + KV-cache

Usage:
  python scripts/week_multimodal.py
  python scripts/week_multimodal.py --model phi-3.5-vision
  python scripts/week_multimodal.py --model llava-1.5-7b
"""

import os
import sys
import json
import gc
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODEL_CANDIDATES, MULTIMODAL_MODELS, BENCHMARK_CFG,
    RESULTS_DIR, PROMPTS_DIR, MULTIMODAL_PROMPTS,
)
from src.utils import setup_logging, save_results_json, set_seed
from src.multimodal_loader import (
    load_multimodal_model_4bit,
    benchmark_multimodal_model,
    download_test_image,
    run_multimodal_inference,
)
from src.kv_cache_quant import create_quantized_kv_cache

import torch

logger = setup_logging("week_multimodal")

WEEK_DIR = os.path.join(RESULTS_DIR, "week_multimodal")


def run_attention_benchmarks(model_key, out_dir):
    """Benchmark eager, SDPA, and FlashAttention-2 for a multimodal model."""
    logger.info(f"Running attention benchmarks for {model_key}...")

    image = download_test_image()
    prompts = MULTIMODAL_PROMPTS
    results = {}

    for attn_impl, config_id in [
        ("eager", "mm_eager"),
        ("sdpa", "mm_sdpa"),
        ("flash_attention_2", "mm_flash_attention_2"),
    ]:
        logger.info(f"  Config: {config_id} (attn={attn_impl})")

        try:
            model, processor = load_multimodal_model_4bit(
                model_key, attn_implementation=attn_impl,
            )
        except Exception as e:
            logger.warning(f"  Could not load with {attn_impl}: {e}")
            results[config_id] = {"status": "failed", "error": str(e)}
            continue

        try:
            bench = benchmark_multimodal_model(
                model=model, processor=processor, model_key=model_key,
                prompts=prompts, image=image,
                config_id=config_id,
                attn_implementation=attn_impl,
                max_new_tokens=BENCHMARK_CFG.output_limit,
                num_warmup=BENCHMARK_CFG.num_warmup_runs,
                num_runs=BENCHMARK_CFG.num_benchmark_runs,
            )
            results[config_id] = bench
        except Exception as e:
            logger.warning(f"  Benchmark failed for {config_id}: {e}")
            results[config_id] = {"status": "failed", "error": str(e)}
        finally:
            del model, processor
            gc.collect()
            torch.cuda.empty_cache()

    save_results_json(results, os.path.join(out_dir, "attention_benchmarks.json"))
    return results


def run_kv_cache_benchmarks(model_key, out_dir):
    """Benchmark KV-cache quantization for a multimodal model."""
    logger.info(f"Running KV-cache benchmarks for {model_key}...")

    image = download_test_image()
    prompts = MULTIMODAL_PROMPTS
    results = {}

    model, processor = load_multimodal_model_4bit(model_key)

    for kv_type in ["fp16", "int4", "int2"]:
        config_id = f"mm_kv_{kv_type}"
        logger.info(f"  Config: {config_id}")

        try:
            bench = benchmark_multimodal_model(
                model=model, processor=processor, model_key=model_key,
                prompts=prompts, image=image,
                config_id=config_id,
                attn_implementation="eager",
                kv_quant_type=kv_type if kv_type != "fp16" else None,
                max_new_tokens=BENCHMARK_CFG.output_limit,
                num_warmup=BENCHMARK_CFG.num_warmup_runs,
                num_runs=BENCHMARK_CFG.num_benchmark_runs,
            )
            results[config_id] = bench
        except Exception as e:
            logger.warning(f"  Benchmark failed for {config_id}: {e}")
            results[config_id] = {"status": "failed", "error": str(e)}

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    save_results_json(results, os.path.join(out_dir, "kv_cache_benchmarks.json"))
    return results


def run_combined_benchmarks(model_key, out_dir):
    """Benchmark combined FlashAttention-2 + KV-cache quantization."""
    logger.info(f"Running combined FA2+KV benchmarks for {model_key}...")

    image = download_test_image()
    prompts = MULTIMODAL_PROMPTS
    results = {}

    try:
        model, processor = load_multimodal_model_4bit(
            model_key, attn_implementation="flash_attention_2",
        )
    except Exception as e:
        logger.warning(f"  FA2 not available for combined test: {e}")
        save_results_json(
            {"status": "fa2_unavailable", "error": str(e)},
            os.path.join(out_dir, "combined_benchmarks.json"),
        )
        return {}

    for kv_type in ["int4", "int2"]:
        config_id = f"mm_fa2_kv_{kv_type}"
        logger.info(f"  Config: {config_id}")

        try:
            bench = benchmark_multimodal_model(
                model=model, processor=processor, model_key=model_key,
                prompts=prompts, image=image,
                config_id=config_id,
                attn_implementation="flash_attention_2",
                kv_quant_type=kv_type,
                max_new_tokens=BENCHMARK_CFG.output_limit,
                num_warmup=BENCHMARK_CFG.num_warmup_runs,
                num_runs=BENCHMARK_CFG.num_benchmark_runs,
            )
            results[config_id] = bench
        except Exception as e:
            logger.warning(f"  Combined benchmark failed for {config_id}: {e}")
            results[config_id] = {"status": "failed", "error": str(e)}

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    save_results_json(results, os.path.join(out_dir, "combined_benchmarks.json"))
    return results


def run_sample_generations(model_key, out_dir):
    """Generate sample outputs for qualitative inspection."""
    logger.info(f"Generating sample outputs for {model_key}...")

    image = download_test_image()
    prompts = MULTIMODAL_PROMPTS[:3]

    model, processor = load_multimodal_model_4bit(model_key)

    samples = []
    for prompt in prompts:
        result = run_multimodal_inference(
            model, processor, model_key, prompt,
            image=image, max_new_tokens=BENCHMARK_CFG.output_limit,
        )
        samples.append({
            "prompt": prompt,
            "status": result.get("status"),
            "generated_text": result.get("generated_text", ""),
            "latency_ms": result.get("total_time_ms", 0),
        })

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    save_results_json(samples, os.path.join(out_dir, "sample_generations.json"))
    return samples


def run_for_model(model_key: str):
    out_dir = os.path.join(WEEK_DIR, model_key)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Attention backends
    attn_results = run_attention_benchmarks(model_key, out_dir)

    # 2. KV-cache quantization
    kv_results = run_kv_cache_benchmarks(model_key, out_dir)

    # 3. Combined FA2 + KV-cache
    combined_results = run_combined_benchmarks(model_key, out_dir)

    # 4. Sample generations
    samples = run_sample_generations(model_key, out_dir)

    # Summary
    all_configs = {}
    all_configs.update(attn_results)
    all_configs.update(kv_results)
    all_configs.update(combined_results)

    successful = [k for k, v in all_configs.items() if v.get("status") == "success"]
    failed = [k for k, v in all_configs.items() if v.get("status") != "success"]

    summary = {
        "week": "multimodal",
        "title": "Multimodal Model Benchmarks",
        "status": "COMPLETE",
        "model_key": model_key,
        "model_id": MODEL_CANDIDATES[model_key]["hf_id"],
        "configs_tested": len(all_configs),
        "configs_successful": len(successful),
        "configs_failed": len(failed),
        "successful_configs": successful,
        "failed_configs": failed,
        "flash_attention_2_tested": any("fa2" in k or "flash" in k for k in all_configs),
        "kv_cache_tested": any("kv" in k for k in all_configs),
        "combined_tested": any("fa2_kv" in k for k in all_configs),
        "deliverables": [
            "attention_benchmarks.json",
            "kv_cache_benchmarks.json",
            "combined_benchmarks.json",
            "sample_generations.json",
        ],
    }
    save_results_json(summary, os.path.join(out_dir, "week_multimodal_summary.json"))

    logger.info(f"  {model_key}: {len(successful)} successful, {len(failed)} failed")
    return summary


def main(model_keys=None):
    from src.utils import CheckpointManager
    ckpt = CheckpointManager("week_multimodal")

    logger.info("=" * 60)
    logger.info("WEEK MULTIMODAL — Vision-Language Model Benchmarks")
    logger.info("=" * 60)

    set_seed(BENCHMARK_CFG.seed)

    if model_keys is None:
        model_keys = list(MULTIMODAL_MODELS)

    for model_key in model_keys:
        if not MODEL_CANDIDATES.get(model_key, {}).get("multimodal", False):
            logger.warning(f"  {model_key} is not a multimodal model, skipping")
            continue

        if ckpt.is_done(model_key):
            logger.info(f"[SKIP] {model_key} already done (checkpoint)")
            continue

        logger.info(f"\n{'#' * 50}")
        logger.info(f"# Multimodal Model: {model_key}")
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
    logger.info("Week Multimodal complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs="+", default=None,
                        help="Multimodal model key(s) to benchmark")
    args = parser.parse_args()
    main(model_keys=args.model)
