"""
Week 05 — Attention Optimization Integration

Deliverables:
  - Attention-optimized configuration scripts and flags
  - Correctness report with numerical metrics
  - Initial attention benchmark results with backend evidence logs
  - Eager-vs-SDPA-vs-FlashAttention-2 comparison (FA2 tested explicitly)

Usage:
  python scripts/week05_attention_opt.py
  python scripts/week05_attention_opt.py --model qwen2.5-3b
"""

import os
import sys
import json
import gc
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODEL_CANDIDATES, PRIMARY_BENCHMARK_MODELS, BENCHMARK_CFG, CORRECTNESS_CFG,
    RESULTS_DIR, PROMPTS_DIR,
)
from src.utils import setup_logging, save_results_json, set_seed
from src.model_loader import load_model_4bit
from src.attention_backends import (
    benchmark_attention_backends,
    benchmark_eager_baseline,
    check_attention_backends,
    sdpa_backend,
)
from src.benchmark_harness import run_benchmark_sweep, results_to_table
from src.correctness import run_correctness_suite

import torch

logger = setup_logging("week05")


def _load_prompts():
    with open(os.path.join(PROMPTS_DIR, "fixed_prompts.json")) as f:
        return json.load(f)


def _load_baseline_traces(model_key):
    """Load baseline correctness traces for comparison."""
    baseline_path = os.path.join(
        RESULTS_DIR, "week03_baseline", model_key, "baseline_correctness_full.json"
    )
    if not os.path.exists(baseline_path):
        baseline_path = os.path.join(
            RESULTS_DIR, "week03_baseline", "baseline_correctness_full.json"
        )
    if os.path.exists(baseline_path):
        from src.utils import load_results_json
        data = load_results_json(baseline_path)
        return data.get("traces", [])
    logger.warning("Baseline correctness traces not found - correctness will only collect, not compare.")
    return None


def _unload_model(model, tokenizer):
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_model_for_attn(model_key, attn_implementation):
    """Load model with the given attn_implementation, using the multimodal loader if needed."""
    model_info = MODEL_CANDIDATES[model_key]
    model_id = model_info["hf_id"]
    is_multimodal = model_info.get("multimodal", False)

    if is_multimodal:
        from src.multimodal_loader import load_multimodal_model_4bit
        model, processor = load_multimodal_model_4bit(
            model_key, attn_implementation=attn_implementation
        )
        tokenizer = getattr(processor, "tokenizer", processor)
    else:
        model, tokenizer = load_model_4bit(model_id, attn_implementation=attn_implementation)
    return model, tokenizer


def _serialize_sweep(sweep_results) -> list:
    return [
        {
            "config_id": r.config_id,
            "prompt_length": r.prompt_length,
            "batch_size": r.batch_size,
            "latency_p50_ms": r.latency_p50_ms,
            "latency_p95_ms": r.latency_p95_ms,
            "throughput_tok_per_s": r.throughput_tokens_per_s,
            "peak_vram_mb": r.peak_vram_mb,
            "oom_occurred": r.oom_occurred,
        }
        for r in sweep_results
    ]


def run_eager_phase(prompts_data, model_id, model_key):
    logger.info("-" * 50)
    logger.info("Phase 1: Eager attention baseline")
    logger.info("-" * 50)

    model, tokenizer = _load_model_for_attn(model_key, attn_implementation="eager")

    prompt = prompts_data["single_turn"][0]["text"]

    eager_bench = benchmark_eager_baseline(
        model, tokenizer, prompt,
        max_new_tokens=BENCHMARK_CFG.output_limit,
        num_runs=BENCHMARK_CFG.num_benchmark_runs,
    )

    correctness_prompts = [
        p["text"]
        for p in prompts_data["single_turn"][:CORRECTNESS_CFG.num_fixed_prompts]
    ]
    baseline_traces = _load_baseline_traces(model_key)

    eager_corr = run_correctness_suite(
        model=model, tokenizer=tokenizer,
        prompts=correctness_prompts,
        baseline_traces=baseline_traces,
        num_steps=CORRECTNESS_CFG.num_logit_comparison_steps,
        config_id="eager", seed=BENCHMARK_CFG.seed,
    )

    eager_sweep = run_benchmark_sweep(
        model=model, tokenizer=tokenizer,
        base_prompt=prompt, config_id="eager",
        prompt_lengths=BENCHMARK_CFG.prompt_lengths,
        output_limit=BENCHMARK_CFG.output_limit,
        batch_sizes=[1],
        num_warmup=BENCHMARK_CFG.num_warmup_runs,
        num_runs=BENCHMARK_CFG.num_benchmark_runs,
        seed=BENCHMARK_CFG.seed,
    )

    _unload_model(model, tokenizer)
    return eager_bench, eager_corr, eager_sweep


def run_sdpa_phase(prompts_data, model_id, model_key):
    logger.info("-" * 50)
    logger.info("Phase 2: SDPA attention backends")
    logger.info("-" * 50)

    model, tokenizer = _load_model_for_attn(model_key, attn_implementation="sdpa")

    prompt = prompts_data["single_turn"][0]["text"]

    available = check_attention_backends()

    backends_to_test: list = ["default"]
    if available.get("flash_sdp_available"):
        backends_to_test.append("flash")
    if available.get("mem_efficient_sdp_available"):
        backends_to_test.append("mem_efficient")
    backends_to_test.append("math")

    backend_bench = benchmark_attention_backends(
        model=model, tokenizer=tokenizer, prompt=prompt,
        backends_to_test=backends_to_test,
        max_new_tokens=BENCHMARK_CFG.output_limit,
        num_runs=BENCHMARK_CFG.num_benchmark_runs,
    )

    correctness_prompts = [
        p["text"]
        for p in prompts_data["single_turn"][:CORRECTNESS_CFG.num_fixed_prompts]
    ]
    baseline_traces = _load_baseline_traces(model_key)

    correctness_results = {}
    for bname in backend_bench:
        if backend_bench[bname].get("status") != "success":
            continue
        logger.info(f"Correctness check: sdpa_{bname}")
        try:
            with sdpa_backend(bname):
                corr = run_correctness_suite(
                    model=model, tokenizer=tokenizer,
                    prompts=correctness_prompts,
                    baseline_traces=baseline_traces,
                    num_steps=CORRECTNESS_CFG.num_logit_comparison_steps,
                    config_id=f"sdpa_{bname}", seed=BENCHMARK_CFG.seed,
                )
            correctness_results[bname] = {
                "config_id": corr.get("config_id"),
                "num_prompts": corr.get("num_prompts"),
                "aggregate": corr.get("aggregate", {}),
            }
        except Exception as e:
            logger.warning(f"Correctness failed for sdpa_{bname}: {e}")
            correctness_results[bname] = {
                "config_id": f"sdpa_{bname}", "status": "failed", "error": str(e),
            }

    sweep_results = {}
    for bname in backend_bench:
        if backend_bench[bname].get("status") != "success":
            continue
        logger.info(f"Benchmark sweep: sdpa_{bname}")
        try:
            with sdpa_backend(bname):
                sweep = run_benchmark_sweep(
                    model=model, tokenizer=tokenizer,
                    base_prompt=prompt, config_id=f"sdpa_{bname}",
                    prompt_lengths=BENCHMARK_CFG.prompt_lengths,
                    output_limit=BENCHMARK_CFG.output_limit,
                    batch_sizes=[1],
                    num_warmup=BENCHMARK_CFG.num_warmup_runs,
                    num_runs=BENCHMARK_CFG.num_benchmark_runs,
                    seed=BENCHMARK_CFG.seed,
                )
            sweep_results[bname] = _serialize_sweep(sweep)
        except Exception as e:
            logger.warning(f"Sweep failed for sdpa_{bname}: {e}")
            sweep_results[bname] = [{"status": "failed", "error": str(e)}]

    _unload_model(model, tokenizer)
    return backend_bench, correctness_results, sweep_results


def run_fa2_phase(prompts_data, model_id, model_key):
    """Explicitly test FlashAttention-2 via flash-attn package."""
    logger.info("-" * 50)
    logger.info("Phase 3: FlashAttention-2 (flash_attn package — explicit)")
    logger.info("-" * 50)

    try:
        model, tokenizer = _load_model_for_attn(model_key, attn_implementation="flash_attention_2")
    except Exception as e:
        logger.warning(f"Could not load model with flash_attention_2: {e}")
        return (
            {"status": "failed", "errors": [str(e)]},
            {"config_id": "flash_attention_2", "status": "failed", "error": str(e)},
            [],
        )

    prompt = prompts_data["single_turn"][0]["text"]

    fa2_bench = benchmark_eager_baseline(
        model, tokenizer, prompt,
        max_new_tokens=BENCHMARK_CFG.output_limit,
        num_runs=BENCHMARK_CFG.num_benchmark_runs,
    )

    correctness_prompts = [
        p["text"]
        for p in prompts_data["single_turn"][:CORRECTNESS_CFG.num_fixed_prompts]
    ]
    baseline_traces = _load_baseline_traces(model_key)

    fa2_corr = run_correctness_suite(
        model=model, tokenizer=tokenizer,
        prompts=correctness_prompts,
        baseline_traces=baseline_traces,
        num_steps=CORRECTNESS_CFG.num_logit_comparison_steps,
        config_id="flash_attention_2", seed=BENCHMARK_CFG.seed,
    )

    fa2_sweep = run_benchmark_sweep(
        model=model, tokenizer=tokenizer,
        base_prompt=prompt, config_id="flash_attention_2",
        prompt_lengths=BENCHMARK_CFG.prompt_lengths,
        output_limit=BENCHMARK_CFG.output_limit,
        batch_sizes=[1],
        num_warmup=BENCHMARK_CFG.num_warmup_runs,
        num_runs=BENCHMARK_CFG.num_benchmark_runs,
        seed=BENCHMARK_CFG.seed,
    )

    _unload_model(model, tokenizer)
    return fa2_bench, fa2_corr, fa2_sweep


def _log_comparison_table(all_bench):
    header = f"{'Backend':<20} {'p50 ms':>10} {'p95 ms':>10} {'tok/s':>10} {'VRAM MB':>10} {'Status':>8}"
    logger.info("")
    logger.info("=" * len(header))
    logger.info("Backend comparison (single prompt)")
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))
    for name, data in all_bench.items():
        if data.get("status") == "success":
            logger.info(
                f"{name:<20} {data['latency_p50_ms']:>10.1f} "
                f"{data['latency_p95_ms']:>10.1f} "
                f"{data['throughput_tok_per_s']:>10.1f} "
                f"{data['peak_vram_mb']:>10.1f} "
                f"{'OK':>8}"
            )
        else:
            logger.info(f"{name:<20} {'-':>10} {'-':>10} {'-':>10} {'-':>10} {'FAIL':>8}")
    logger.info("=" * len(header))


def run_for_model(model_key: str):
    """Run all week-05 phases for a single model."""
    model_info = MODEL_CANDIDATES[model_key]
    model_id = model_info["hf_id"]

    out_dir = os.path.join(RESULTS_DIR, "week05_attention_opt", model_key)
    os.makedirs(out_dir, exist_ok=True)

    prompts_data = _load_prompts()

    # Phase 1: eager
    eager_bench, eager_corr, eager_sweep = run_eager_phase(prompts_data, model_id, model_key)

    # Phase 2: SDPA
    backend_bench, sdpa_correctness, sdpa_sweeps = run_sdpa_phase(prompts_data, model_id, model_key)

    # Phase 3: FlashAttention-2 (always attempted)
    logger.info("Attempting FlashAttention-2 explicitly (not just SDPA fallback)...")
    fa2_bench, fa2_corr, fa2_sweep = run_fa2_phase(prompts_data, model_id, model_key)

    # Merge benchmarks
    all_bench = {"eager": eager_bench}
    for bname, data in backend_bench.items():
        all_bench[f"sdpa_{bname}"] = data
    all_bench["flash_attention_2"] = fa2_bench

    save_results_json(all_bench, os.path.join(out_dir, "attention_backend_benchmark.json"))

    # Merge correctness
    all_correctness = {
        "eager": {
            "config_id": eager_corr.get("config_id"),
            "num_prompts": eager_corr.get("num_prompts"),
            "aggregate": eager_corr.get("aggregate", {}),
        },
    }
    all_correctness.update(sdpa_correctness)
    if isinstance(fa2_corr, dict) and fa2_corr.get("config_id"):
        all_correctness["flash_attention_2"] = {
            "config_id": fa2_corr.get("config_id"),
            "num_prompts": fa2_corr.get("num_prompts"),
            "aggregate": fa2_corr.get("aggregate", {}),
        }
    save_results_json(all_correctness, os.path.join(out_dir, "attention_correctness_report.json"))

    # Merge sweeps
    all_sweeps = {"eager": _serialize_sweep(eager_sweep)}
    all_sweeps.update(sdpa_sweeps)
    if isinstance(fa2_sweep, list) and fa2_sweep:
        all_sweeps["flash_attention_2"] = _serialize_sweep(fa2_sweep) if hasattr(fa2_sweep[0], "config_id") else fa2_sweep
    elif hasattr(fa2_sweep, "__iter__") and not isinstance(fa2_sweep, dict):
        all_sweeps["flash_attention_2"] = _serialize_sweep(fa2_sweep)
    save_results_json(all_sweeps, os.path.join(out_dir, "attention_benchmark_sweeps.json"))

    _log_comparison_table(all_bench)

    summary = {
        "week": "05",
        "title": "Attention Optimization Integration",
        "status": "COMPLETE",
        "model_key": model_key,
        "model": model_id,
        "backends_tested": list(all_bench.keys()),
        "backends_successful": [
            k for k, v in all_bench.items() if v.get("status") == "success"
        ],
        "flash_attention_2_tested": True,
        "deliverables": [
            "attention_backend_benchmark.json",
            "attention_correctness_report.json",
            "attention_benchmark_sweeps.json",
        ],
    }
    save_results_json(summary, os.path.join(out_dir, "week05_summary.json"))


def main(model_keys=None):
    from src.utils import CheckpointManager
    ckpt = CheckpointManager("week05")

    logger.info("=" * 60)
    logger.info("WEEK 05 — Attention Optimization Integration")
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
    logger.info("Week 05 complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs="+", default=None)
    args = parser.parse_args()
    main(model_keys=args.model)
