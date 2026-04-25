"""
Week 06 — Attention Optimization Stabilization & Final Results

Deliverables:
  - Final attention results table and plots: latency vs length, VRAM vs length, tokens/s
  - OOM thresholds per working backend (including FlashAttention-2)
  - Stability notes and limitations section

Usage:
  python scripts/week06_attention_final.py
  python scripts/week06_attention_final.py --model qwen2.5-3b
"""

import os
import sys
import json
import gc
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODEL_CANDIDATES, PRIMARY_BENCHMARK_MODELS, BENCHMARK_CFG,
    RESULTS_DIR, PROMPTS_DIR,
)
from src.utils import setup_logging, save_results_json, save_results_csv, set_seed
from src.model_loader import load_model_4bit
from src.attention_backends import sdpa_backend, check_attention_backends
from src.benchmark_harness import (
    run_benchmark_sweep, find_oom_threshold, results_to_table,
)

import torch

logger = setup_logging("week06")


def _unload(model, tokenizer):
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def run_complete_attention_benchmarks(model_key):
    """Run full benchmark sweeps for all working attention configs."""
    logger.info("Running complete attention benchmark sweeps...")

    with open(os.path.join(PROMPTS_DIR, "fixed_prompts.json")) as f:
        prompts = json.load(f)
    base_prompt = prompts["single_turn"][0]["text"]
    available = check_attention_backends()

    all_results = []

    # 1. Eager baseline
    logger.info("--- Eager (no attention optimization) ---")
    model, tokenizer = _load_model_for_attn(model_key, attn_implementation="eager")
    try:
        results = run_benchmark_sweep(
            model=model, tokenizer=tokenizer, base_prompt=base_prompt,
            config_id="eager",
            prompt_lengths=BENCHMARK_CFG.prompt_lengths,
            output_limit=BENCHMARK_CFG.output_limit,
            batch_sizes=BENCHMARK_CFG.batch_sizes,
            num_warmup=BENCHMARK_CFG.num_warmup_runs,
            num_runs=BENCHMARK_CFG.num_benchmark_runs,
            seed=BENCHMARK_CFG.seed,
        )
        all_results.extend(results)
    finally:
        _unload(model, tokenizer)

    # 2. SDPA Default
    logger.info("--- SDPA Default ---")
    model, tokenizer = _load_model_for_attn(model_key, attn_implementation="sdpa")
    try:
        results = run_benchmark_sweep(
            model=model, tokenizer=tokenizer, base_prompt=base_prompt,
            config_id="sdpa_default",
            prompt_lengths=BENCHMARK_CFG.prompt_lengths,
            output_limit=BENCHMARK_CFG.output_limit,
            batch_sizes=BENCHMARK_CFG.batch_sizes,
            num_warmup=BENCHMARK_CFG.num_warmup_runs,
            num_runs=BENCHMARK_CFG.num_benchmark_runs,
            seed=BENCHMARK_CFG.seed,
        )
        all_results.extend(results)

        # 3. SDPA Math (same model, kernel override)
        logger.info("--- SDPA Math ---")
        with sdpa_backend("math"):
            results = run_benchmark_sweep(
                model=model, tokenizer=tokenizer, base_prompt=base_prompt,
                config_id="sdpa_math",
                prompt_lengths=BENCHMARK_CFG.prompt_lengths,
                output_limit=BENCHMARK_CFG.output_limit,
                batch_sizes=BENCHMARK_CFG.batch_sizes,
                num_warmup=BENCHMARK_CFG.num_warmup_runs,
                num_runs=BENCHMARK_CFG.num_benchmark_runs,
                seed=BENCHMARK_CFG.seed,
            )
        all_results.extend(results)
    finally:
        _unload(model, tokenizer)

    # 4. FlashAttention-2 (always attempted explicitly)
    logger.info("--- FlashAttention-2 (explicit) ---")
    try:
        model, tokenizer = _load_model_for_attn(model_key, attn_implementation="flash_attention_2")
        try:
            results = run_benchmark_sweep(
                model=model, tokenizer=tokenizer, base_prompt=base_prompt,
                config_id="flash_attention_2",
                prompt_lengths=BENCHMARK_CFG.prompt_lengths,
                output_limit=BENCHMARK_CFG.output_limit,
                batch_sizes=BENCHMARK_CFG.batch_sizes,
                num_warmup=BENCHMARK_CFG.num_warmup_runs,
                num_runs=BENCHMARK_CFG.num_benchmark_runs,
                seed=BENCHMARK_CFG.seed,
            )
            all_results.extend(results)
        finally:
            _unload(model, tokenizer)
    except Exception as e:
        logger.warning(f"FlashAttention-2 not available: {e}")

    return all_results


def run_oom_thresholds(model_key):
    """Find OOM threshold for each working backend."""
    with open(os.path.join(PROMPTS_DIR, "fixed_prompts.json")) as f:
        prompts = json.load(f)
    base_long = prompts["long_context_seed"]["text"]

    oom_results = {}

    # Eager
    logger.info("OOM threshold: eager...")
    model, tokenizer = _load_model_for_attn(model_key, attn_implementation="eager")
    try:
        oom = find_oom_threshold(
            model=model, tokenizer=tokenizer,
            base_prompt=base_long,
            max_new_tokens=BENCHMARK_CFG.output_limit,
            start_length=128, max_length=8192, step=256,
        )
        oom_results["eager"] = oom
    finally:
        _unload(model, tokenizer)

    # SDPA Default
    logger.info("OOM threshold: sdpa_default...")
    model, tokenizer = _load_model_for_attn(model_key, attn_implementation="sdpa")
    try:
        oom = find_oom_threshold(
            model=model, tokenizer=tokenizer,
            base_prompt=base_long,
            max_new_tokens=BENCHMARK_CFG.output_limit,
            start_length=128, max_length=8192, step=256,
        )
        oom_results["sdpa_default"] = oom
    finally:
        _unload(model, tokenizer)

    # FlashAttention-2 (always attempted)
    logger.info("OOM threshold: flash_attention_2 (explicit)...")
    try:
        model, tokenizer = _load_model_for_attn(model_key, attn_implementation="flash_attention_2")
        try:
            oom = find_oom_threshold(
                model=model, tokenizer=tokenizer,
                base_prompt=base_long,
                max_new_tokens=BENCHMARK_CFG.output_limit,
                start_length=128, max_length=8192, step=256,
            )
            oom_results["flash_attention_2"] = oom
        finally:
            _unload(model, tokenizer)
    except Exception as e:
        logger.warning(f"FlashAttention-2 OOM test skipped: {e}")

    return oom_results


def generate_stability_notes(all_results, oom_results):
    failure_modes = []
    for r in all_results:
        if r.oom_occurred:
            failure_modes.append(
                f"OOM: config={r.config_id}, prompt_len={r.prompt_length}, "
                f"batch_size={r.batch_size}"
            )

    notes = {
        "thermal_throttling": (
            "Laptop GPUs may throttle under sustained load. A 1-2 minute warmup "
            "and monitoring GPU temperature is recommended."
        ),
        "backend_availability": {
            "flash_sdp": "Not compiled in PyTorch Windows wheel",
            "mem_efficient_sdp": "May fail on GQA models (head mismatch)",
            "sdpa_math": "Universal fallback, always works",
            "flash_attention_2": "Requires flash-attn package; tested explicitly",
        },
        "oom_thresholds": {
            k: v.get("oom_threshold_tokens", "N/A")
            for k, v in oom_results.items()
        },
        "failure_modes_observed": failure_modes,
    }
    return notes


def run_for_model(model_key: str):
    model_info = MODEL_CANDIDATES[model_key]
    model_id = model_info["hf_id"]

    out_dir = os.path.join(RESULTS_DIR, "week06_attention_final", model_key)
    os.makedirs(out_dir, exist_ok=True)

    all_results = run_complete_attention_benchmarks(model_key)
    oom_results = run_oom_thresholds(model_key)

    table_rows = results_to_table(all_results)
    save_results_csv(table_rows, os.path.join(out_dir, "attention_final_results.csv"))
    save_results_json(table_rows, os.path.join(out_dir, "attention_final_results.json"))
    save_results_json(oom_results, os.path.join(out_dir, "attention_oom_thresholds.json"))

    stability_notes = generate_stability_notes(all_results, oom_results)
    save_results_json(stability_notes, os.path.join(out_dir, "stability_notes.json"))

    summary = {
        "week": "06",
        "title": "Attention Optimization Final Results",
        "status": "COMPLETE",
        "model_key": model_key,
        "model": model_id,
        "num_configurations_tested": len(set(r.config_id for r in all_results)),
        "total_benchmark_runs": sum(r.num_runs for r in all_results),
        "oom_thresholds": {
            k: v.get("oom_threshold_tokens") for k, v in oom_results.items()
        },
        "flash_attention_2_tested": "flash_attention_2" in {r.config_id for r in all_results},
        "deliverables": [
            "attention_final_results.csv",
            "attention_final_results.json",
            "attention_oom_thresholds.json",
            "stability_notes.json",
        ],
    }
    save_results_json(summary, os.path.join(out_dir, "week06_summary.json"))


def main(model_keys=None):
    from src.utils import CheckpointManager
    ckpt = CheckpointManager("week06")

    logger.info("=" * 60)
    logger.info("WEEK 06 — Attention Optimization Final Results")
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
    logger.info("Week 06 complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs="+", default=None)
    args = parser.parse_args()
    main(model_keys=args.model)
