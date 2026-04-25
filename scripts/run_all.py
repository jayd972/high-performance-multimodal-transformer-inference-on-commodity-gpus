"""
Master runner — Executes all weekly scripts sequentially with checkpointing.

Checkpoints are saved per (week, model) so if your laptop crashes, re-running
the same command will skip already-completed work.

Usage:
    python scripts/run_all.py                         # Run all weeks, all models
    python scripts/run_all.py --week 3                # Run specific week
    python scripts/run_all.py --from 5 --to 8         # Run weeks 5-8
    python scripts/run_all.py --model qwen2.5-3b      # Run all weeks for one model
    python scripts/run_all.py --multimodal             # Run multimodal benchmarks only
    python scripts/run_all.py --plots                  # Generate plots after running
    python scripts/run_all.py --reset-checkpoints      # Clear all checkpoints and start fresh
"""

import os
import sys
import argparse
import importlib
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import CheckpointManager

SCRIPTS = {
    1: ("week01_planning", "Planning & Scope Definition"),
    2: ("week02_setup_validation", "Environment Setup & Validation"),
    3: ("week03_baseline", "Baseline Measurement"),
    4: ("week04_profiling", "Profiling & Attention Backend Planning"),
    5: ("week05_attention_opt", "Attention Optimization Integration"),
    6: ("week06_attention_final", "Attention Optimization Final Results"),
    7: ("week07_kv_cache", "KV-Cache Quantization Phase"),
    8: ("week08_kv_experiments", "KV-Cache Experiments & Quality Evaluation"),
    9: ("week09_combined", "Combined Configuration"),
    10: ("week10_full_benchmark", "Full Benchmark Suite"),
}

WEEKS_WITH_MODEL_SUPPORT = {3, 5, 6, 7, 8, 9, 10}

ckpt = CheckpointManager("run_all")


def run_week(week_num: int, model_keys=None) -> bool:
    """Run a specific week's script with checkpoint awareness."""
    if week_num not in SCRIPTS:
        print(f"Week {week_num} not found.")
        return False

    ckpt_key = f"week{week_num:02d}"
    if model_keys:
        ckpt_key += f"_{'_'.join(sorted(model_keys))}"

    if ckpt.is_done(ckpt_key):
        print(f"  [SKIP] Week {week_num:02d} already completed (checkpoint found)")
        return True

    module_name, title = SCRIPTS[week_num]
    print(f"\n{'#' * 60}")
    print(f"# WEEK {week_num:02d} — {title}")
    print(f"{'#' * 60}\n")

    try:
        start = time.time()
        module = importlib.import_module(f"scripts.{module_name}")

        if model_keys and week_num in WEEKS_WITH_MODEL_SUPPORT:
            module.main(model_keys=model_keys)
        else:
            module.main()

        elapsed = time.time() - start
        ckpt.mark_done(ckpt_key)
        print(f"\nWeek {week_num:02d} completed in {elapsed:.1f}s  [CHECKPOINT SAVED]\n")
        return True
    except Exception as e:
        print(f"\nWeek {week_num:02d} FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def run_multimodal(model_keys=None) -> bool:
    """Run multimodal-specific benchmarks with checkpointing."""
    ckpt_key = "multimodal"
    if model_keys:
        ckpt_key += f"_{'_'.join(sorted(model_keys))}"

    if ckpt.is_done(ckpt_key):
        print(f"  [SKIP] Multimodal already completed (checkpoint found)")
        return True

    print(f"\n{'#' * 60}")
    print(f"# MULTIMODAL — Vision-Language Model Benchmarks")
    print(f"{'#' * 60}\n")

    try:
        start = time.time()
        module = importlib.import_module("scripts.week_multimodal")
        module.main(model_keys=model_keys)
        elapsed = time.time() - start
        ckpt.mark_done(ckpt_key)
        print(f"\nMultimodal benchmarks completed in {elapsed:.1f}s  [CHECKPOINT SAVED]\n")
        return True
    except Exception as e:
        print(f"\nMultimodal benchmarks FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Run weekly experiment scripts")
    parser.add_argument("--week", type=int, help="Run a specific week (1-10)")
    parser.add_argument("--from", dest="from_week", type=int, default=1,
                        help="Start from this week (default: 1)")
    parser.add_argument("--to", type=int, default=10,
                        help="End at this week (default: 10)")
    parser.add_argument("--model", type=str, nargs="+", default=None,
                        help="Model key(s) to benchmark (default: all)")
    parser.add_argument("--multimodal", action="store_true",
                        help="Run multimodal benchmarks")
    parser.add_argument("--plots", action="store_true",
                        help="Generate plots after running")
    parser.add_argument("--reset-checkpoints", action="store_true",
                        help="Clear all checkpoints and start fresh")
    args = parser.parse_args()

    if args.reset_checkpoints:
        ckpt.reset()
        print("All checkpoints cleared.")

    total_start = time.time()
    results = {}

    if args.multimodal:
        success = run_multimodal(model_keys=args.model)
        results["multimodal"] = success
    elif args.week:
        success = run_week(args.week, model_keys=args.model)
        results[args.week] = success
    else:
        for week_num in range(args.from_week, args.to + 1):
            if week_num in SCRIPTS:
                success = run_week(week_num, model_keys=args.model)
                results[week_num] = success
                if not success:
                    print(f"Week {week_num} failed — continuing to next week...")

        if args.model is None or any(
            m in (args.model or [])
            for m in ["phi-3.5-vision", "llava-1.5-7b"]
        ):
            mm_keys = args.model if args.model else None
            success = run_multimodal(model_keys=mm_keys)
            results["multimodal"] = success

    if args.plots:
        print("\nGenerating plots...")
        try:
            from scripts.generate_plots import main as gen_plots
            gen_plots()
        except Exception as e:
            print(f"Plot generation failed: {e}")

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 60}")
    print("EXECUTION SUMMARY")
    print(f"{'=' * 60}")
    for key, success in results.items():
        status = "PASS" if success else "FAIL"
        if isinstance(key, int):
            _, title = SCRIPTS[key]
            print(f"  Week {key:02d} ({title}): {status}")
        else:
            print(f"  {key}: {status}")
    if args.model:
        print(f"\nModel(s): {', '.join(args.model)}")
    print(f"\nCheckpoint file: {ckpt._path}")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
