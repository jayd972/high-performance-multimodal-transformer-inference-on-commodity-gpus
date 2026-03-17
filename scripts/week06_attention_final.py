"""
Week 06 — Attention Optimization Stabilization & Final Results

Deliverables:
  - Final attention results table and plots: latency vs length, VRAM vs length, tokens/s
  - OOM thresholds per working backend
  - Stability notes and limitations section
  - Report draft subsection text with figure placeholders

Uses only backends that work on this setup:
  - eager (no attention optimization)
  - sdpa_default, sdpa_math (PyTorch SDPA)
  - flash_attention_2 (flash-attn package)
Skips: flash SDP (not in PyTorch Windows wheel), mem_efficient SDP (GQA incompatible)
"""

import os
import sys
import json
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_MODEL_ID, BENCHMARK_CFG, RESULTS_DIR, PROMPTS_DIR
from src.utils import setup_logging, save_results_json, save_results_csv, set_seed
from src.model_loader import load_model_4bit
from src.attention_backends import sdpa_backend, check_attention_backends
from src.benchmark_harness import (
    run_benchmark_sweep, find_oom_threshold, results_to_table,
)

import torch

logger = setup_logging("llm_inference.week06")

WEEK_DIR = os.path.join(RESULTS_DIR, "week06_attention_final")
os.makedirs(WEEK_DIR, exist_ok=True)


def _unload(model, tokenizer):
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_complete_attention_benchmarks():
    """Run full benchmark sweeps for all working attention configs."""
    logger.info("Running complete attention benchmark sweeps...")

    with open(os.path.join(PROMPTS_DIR, "fixed_prompts.json")) as f:
        prompts = json.load(f)
    base_prompt = prompts["single_turn"][0]["text"]
    available = check_attention_backends()

    all_results = []

    # 1. Eager baseline
    logger.info("--- Eager (no attention optimization) ---")
    model, tokenizer = load_model_4bit(DEFAULT_MODEL_ID, attn_implementation="eager")
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
    model, tokenizer = load_model_4bit(DEFAULT_MODEL_ID, attn_implementation="sdpa")
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

    # 4. FlashAttention-2
    if available.get("flash_attn_2_available"):
        logger.info("--- FlashAttention-2 ---")
        model, tokenizer = load_model_4bit(
            DEFAULT_MODEL_ID, attn_implementation="flash_attention_2"
        )
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
    else:
        logger.info("FlashAttention-2 not available - skipping")

    return all_results


def run_oom_thresholds():
    """Find OOM threshold for each working backend."""
    with open(os.path.join(PROMPTS_DIR, "fixed_prompts.json")) as f:
        prompts = json.load(f)
    base_long = prompts["long_context_seed"]["text"]
    available = check_attention_backends()

    oom_results = {}

    # Eager
    logger.info("OOM threshold: eager...")
    model, tokenizer = load_model_4bit(DEFAULT_MODEL_ID, attn_implementation="eager")
    try:
        oom = find_oom_threshold(
            model=model, tokenizer=tokenizer,
            base_prompt=base_long,
            max_new_tokens=BENCHMARK_CFG.output_limit,
            start_length=512, max_length=8192, step=512,
        )
        oom_results["eager"] = oom
    finally:
        _unload(model, tokenizer)

    # SDPA Default
    logger.info("OOM threshold: sdpa_default...")
    model, tokenizer = load_model_4bit(DEFAULT_MODEL_ID, attn_implementation="sdpa")
    try:
        oom = find_oom_threshold(
            model=model, tokenizer=tokenizer,
            base_prompt=base_long,
            max_new_tokens=BENCHMARK_CFG.output_limit,
            start_length=512, max_length=8192, step=512,
        )
        oom_results["sdpa_default"] = oom
    finally:
        _unload(model, tokenizer)

    # FlashAttention-2
    if available.get("flash_attn_2_available"):
        logger.info("OOM threshold: flash_attention_2...")
        model, tokenizer = load_model_4bit(
            DEFAULT_MODEL_ID, attn_implementation="flash_attention_2"
        )
        try:
            oom = find_oom_threshold(
                model=model, tokenizer=tokenizer,
                base_prompt=base_long,
                max_new_tokens=BENCHMARK_CFG.output_limit,
                start_length=512, max_length=8192, step=512,
            )
            oom_results["flash_attention_2"] = oom
        finally:
            _unload(model, tokenizer)

    return oom_results


def generate_stability_notes(all_results, oom_results):
    """Document stability issues and limitations."""
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
            "and monitoring GPU temperature is recommended. If p95 latency is more "
            "than 2x p50, suspect thermal throttling."
        ),
        "backend_availability": {
            "flash_sdp": "Not compiled in PyTorch Windows wheel",
            "mem_efficient_sdp": "Fails on Qwen2.5-3B (GQA: 16Q vs 2KV head mismatch)",
            "sdpa_math": "Universal fallback, always works",
            "flash_attention_2": "Works via flash-attn package (pre-built wheel)",
        },
        "oom_thresholds": {
            k: v.get("oom_threshold_tokens", "N/A")
            for k, v in oom_results.items()
        },
        "failure_modes_observed": failure_modes,
    }
    return notes


def main():
    logger.info("=" * 60)
    logger.info("WEEK 06 — Attention Optimization Final Results")
    logger.info("=" * 60)

    set_seed(BENCHMARK_CFG.seed)

    # 1. Complete benchmarks
    all_results = run_complete_attention_benchmarks()

    # 2. OOM thresholds
    oom_results = run_oom_thresholds()

    # 3. Save results
    table_rows = results_to_table(all_results)
    save_results_csv(table_rows, os.path.join(WEEK_DIR, "attention_final_results.csv"))
    save_results_json(table_rows, os.path.join(WEEK_DIR, "attention_final_results.json"))
    save_results_json(oom_results, os.path.join(WEEK_DIR, "attention_oom_thresholds.json"))

    # 4. Stability notes
    stability_notes = generate_stability_notes(all_results, oom_results)
    save_results_json(stability_notes, os.path.join(WEEK_DIR, "stability_notes.json"))

    # 5. Summary
    summary = {
        "week": "06",
        "title": "Attention Optimization Final Results",
        "status": "COMPLETE",
        "model": DEFAULT_MODEL_ID,
        "num_configurations_tested": len(set(r.config_id for r in all_results)),
        "total_benchmark_runs": sum(r.num_runs for r in all_results),
        "oom_thresholds": {
            k: v.get("oom_threshold_tokens") for k, v in oom_results.items()
        },
        "deliverables": [
            "attention_final_results.csv",
            "attention_final_results.json",
            "attention_oom_thresholds.json",
            "stability_notes.json",
        ],
    }
    save_results_json(summary, os.path.join(WEEK_DIR, "week06_summary.json"))

    logger.info("=" * 60)
    logger.info("Week 06 deliverables saved to: " + WEEK_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
