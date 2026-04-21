"""
Master runner — Executes all weekly scripts sequentially.

Usage:
    python scripts/run_all.py              # Run all weeks
    python scripts/run_all.py --week 3     # Run specific week
    python scripts/run_all.py --from 5     # Run from week 5 onward
"""

import os
import sys
import argparse
import importlib
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
# Weeks 11–14 are documentation/report phases (no runnable scripts).
# Results are in results/week11_packaging/ through results/week14_final/.


def run_week(week_num: int) -> bool:
    """Run a specific week's script."""
    if week_num not in SCRIPTS:
        print(f"Week {week_num} not found.")
        return False

    module_name, title = SCRIPTS[week_num]
    print(f"\n{'#' * 60}")
    print(f"# WEEK {week_num:02d} — {title}")
    print(f"{'#' * 60}\n")

    try:
        start = time.time()
        module = importlib.import_module(f"scripts.{module_name}")
        module.main()
        elapsed = time.time() - start
        print(f"\nWeek {week_num:02d} completed in {elapsed:.1f}s\n")
        return True
    except Exception as e:
        print(f"\nWeek {week_num:02d} FAILED: {e}\n")
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
    parser.add_argument("--plots", action="store_true",
                        help="Generate plots after running")
    args = parser.parse_args()

    total_start = time.time()
    results = {}

    if args.week:
        success = run_week(args.week)
        results[args.week] = success
    else:
        for week_num in range(args.from_week, args.to + 1):
            if week_num in SCRIPTS:
                success = run_week(week_num)
                results[week_num] = success
                if not success:
                    print(f"Stopping at week {week_num} due to failure.")
                    break

    # Generate plots if requested
    if args.plots:
        print("\nGenerating plots...")
        try:
            from scripts.generate_plots import main as gen_plots
            gen_plots()
        except Exception as e:
            print(f"Plot generation failed: {e}")

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'=' * 60}")
    print("EXECUTION SUMMARY")
    print(f"{'=' * 60}")
    for week, success in results.items():
        status = "PASS" if success else "FAIL"
        _, title = SCRIPTS[week]
        print(f"  Week {week:02d} ({title}): {status}")
    print(f"\nTotal time: {total_elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
