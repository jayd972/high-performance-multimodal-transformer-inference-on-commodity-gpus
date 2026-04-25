"""
Plot generation for all weeks.

Generates publication-quality plots from benchmark results:
  - Latency vs sequence length (per config)
  - VRAM vs sequence length (per config)
  - Throughput comparison bar charts
  - OOM threshold comparison
  - Quality retention comparison
  - KV-cache memory per token

Handles model-specific subdirectories under each week's results folder.
"""

import os
import sys
import json
import glob

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import seaborn as sns
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

# Style
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


def load_json(path):
    """Load JSON file if it exists."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _find_result_file(week_dir, filename):
    """Find a result file, checking both top-level and model subdirectories."""
    # Try top-level first
    top = os.path.join(week_dir, filename)
    if os.path.exists(top):
        return [top]

    # Try model subdirectories
    paths = []
    if os.path.isdir(week_dir):
        for entry in os.listdir(week_dir):
            candidate = os.path.join(week_dir, entry, filename)
            if os.path.exists(candidate):
                paths.append(candidate)
    return paths


def _load_from_week(week_dir, filename):
    """Load JSON data from a week dir, merging across model subdirectories."""
    paths = _find_result_file(week_dir, filename)
    if not paths:
        return None, []

    all_data = []
    model_names = []
    for p in paths:
        data = load_json(p)
        if data is not None:
            model_name = os.path.basename(os.path.dirname(p))
            if model_name == os.path.basename(week_dir):
                model_name = "default"
            all_data.append(data)
            model_names.append(model_name)

    if not all_data:
        return None, []

    return all_data, model_names


# ──────────────────────────────────────────────────────────────────────
# Baseline Plots (Week 03)
# ──────────────────────────────────────────────────────────────────────

def plot_baseline_scaling(output_dir):
    """Plot baseline latency and VRAM scaling with sequence length."""
    week_dir = os.path.join(RESULTS_DIR, "week03_baseline")
    datasets, model_names = _load_from_week(week_dir, "baseline_benchmark_results.json")

    if not datasets:
        print("No baseline data found. Skipping baseline plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for data, model_name in zip(datasets, model_names):
        if not isinstance(data, list):
            continue
        df = pd.DataFrame(data)
        suffix = f"_{model_name}" if model_name != "default" else ""

        # Latency vs sequence length (batch_size=1)
        df_bs1 = df[df["batch_size"] == 1] if "batch_size" in df.columns else df
        if not df_bs1.empty:
            fig, ax = plt.subplots()
            ax.plot(df_bs1["prompt_length"], df_bs1["latency_p50_ms"], "o-", label="p50")
            ax.plot(df_bs1["prompt_length"], df_bs1["latency_p95_ms"], "s--", label="p95")
            ax.set_xlabel("Prompt Length (tokens)")
            ax.set_ylabel("Latency (ms)")
            ax.set_title(f"Baseline Latency vs Sequence Length{' — ' + model_name if suffix else ''}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"baseline_latency_vs_length{suffix}.png"))
            plt.close(fig)

        # VRAM vs sequence length
        if not df_bs1.empty and "peak_vram_mb" in df_bs1.columns:
            fig, ax = plt.subplots()
            ax.plot(df_bs1["prompt_length"], df_bs1["peak_vram_mb"], "o-", color="red")
            ax.set_xlabel("Prompt Length (tokens)")
            ax.set_ylabel("Peak VRAM (MB)")
            ax.set_title(f"Baseline Peak VRAM vs Sequence Length{' — ' + model_name if suffix else ''}")
            ax.axhline(y=4096, color="gray", linestyle="--", alpha=0.5, label="4 GB limit")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"baseline_vram_vs_length{suffix}.png"))
            plt.close(fig)

    print(f"Baseline plots saved to {output_dir}")


# ──────────────────────────────────────────────────────────────────────
# Attention Comparison Plots (Week 06)
# ──────────────────────────────────────────────────────────────────────

def plot_attention_comparison(output_dir):
    """Plot attention backend comparison."""
    week_dir = os.path.join(RESULTS_DIR, "week06_attention_final")
    datasets, model_names = _load_from_week(week_dir, "attention_final_results.json")

    if not datasets:
        print("No attention data found. Skipping attention plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for data, model_name in zip(datasets, model_names):
        if not isinstance(data, list):
            continue
        df = pd.DataFrame(data)
        suffix = f"_{model_name}" if model_name != "default" else ""

        df_bs1 = df[df["batch_size"] == 1] if "batch_size" in df.columns else df
        if not df_bs1.empty:
            # Latency comparison
            fig, ax = plt.subplots()
            configs = df_bs1["config_id"].unique()
            x = np.arange(len(df_bs1["prompt_length"].unique()))
            width = 0.8 / max(len(configs), 1)

            for i, config in enumerate(configs):
                subset = df_bs1[df_bs1["config_id"] == config].sort_values("prompt_length")
                ax.bar(
                    x + i * width, subset["latency_p95_ms"],
                    width, label=config, alpha=0.8,
                )

            ax.set_xlabel("Prompt Length (tokens)")
            ax.set_ylabel("p95 Latency (ms)")
            ax.set_title(f"Attention Backend Comparison — p95 Latency{' — ' + model_name if suffix else ''}")
            ax.set_xticks(x + width * (len(configs) - 1) / 2)
            ax.set_xticklabels(sorted(df_bs1["prompt_length"].unique()))
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"attention_latency_comparison{suffix}.png"))
            plt.close(fig)

            # Throughput comparison
            fig, ax = plt.subplots()
            for config in df_bs1["config_id"].unique():
                subset = df_bs1[df_bs1["config_id"] == config].sort_values("prompt_length")
                ax.plot(
                    subset["prompt_length"], subset["throughput_tok_per_s"],
                    "o-", label=config,
                )
            ax.set_xlabel("Prompt Length (tokens)")
            ax.set_ylabel("Throughput (tokens/s)")
            ax.set_title(f"Attention Backend Comparison — Throughput{' — ' + model_name if suffix else ''}")
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"attention_throughput_comparison{suffix}.png"))
            plt.close(fig)

            # VRAM comparison
            fig, ax = plt.subplots()
            for config in df_bs1["config_id"].unique():
                subset = df_bs1[df_bs1["config_id"] == config].sort_values("prompt_length")
                ax.plot(
                    subset["prompt_length"], subset["peak_vram_mb"],
                    "o-", label=config,
                )
            ax.set_xlabel("Prompt Length (tokens)")
            ax.set_ylabel("Peak VRAM (MB)")
            ax.set_title(f"Attention Backend Comparison — Peak VRAM{' — ' + model_name if suffix else ''}")
            ax.axhline(y=4096, color="gray", linestyle="--", alpha=0.5, label="4 GB limit")
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"attention_vram_comparison{suffix}.png"))
            plt.close(fig)

    print(f"Attention comparison plots saved to {output_dir}")


# ──────────────────────────────────────────────────────────────────────
# KV-Cache Plots (Week 08)
# ──────────────────────────────────────────────────────────────────────

def plot_kv_cache_tradeoffs(output_dir):
    """Plot KV-cache tradeoff curves."""
    week08_dir = os.path.join(RESULTS_DIR, "week08_kv_experiments")
    datasets, model_names = _load_from_week(week08_dir, "kv_tradeoff_analysis.json")

    os.makedirs(output_dir, exist_ok=True)

    if datasets:
        for data, model_name in zip(datasets, model_names):
            suffix = f"_{model_name}" if model_name != "default" else ""
            entries = data.get("per_prompt_length", {}) if isinstance(data, dict) else {}
            if entries:
                fig, ax = plt.subplots()
                for key, val in entries.items():
                    ax.scatter(
                        val.get("vram_saved_pct", 0), val.get("latency_change_pct", 0),
                        s=100, label=key, zorder=5,
                    )
                ax.set_xlabel("VRAM Saved (%)")
                ax.set_ylabel("Latency Change (%)")
                ax.set_title(f"KV-Cache Quantization: Memory vs Latency{' — ' + model_name if suffix else ''}")
                ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
                ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, f"kv_vram_vs_latency_tradeoff{suffix}.png"))
                plt.close(fig)
    else:
        print("No KV tradeoff data found. Skipping tradeoff plots.")

    # Memory per token
    week07_dir = os.path.join(RESULTS_DIR, "week07_kv_cache")
    mem_datasets, mem_models = _load_from_week(week07_dir, "kv_memory_per_token.json")
    if mem_datasets:
        for mem_data, model_name in zip(mem_datasets, mem_models):
            suffix = f"_{model_name}" if model_name != "default" else ""
            fig, ax = plt.subplots()
            has_data = False
            for qt, info in mem_data.items():
                measurements = info.get("measurements", [])
                valid = [m for m in measurements if not m.get("oom") and m.get("peak_vram_mb", 0) > 0]
                if valid:
                    lengths = [m["prompt_length"] for m in valid]
                    vrams = [m["peak_vram_mb"] for m in valid]
                    ax.plot(lengths, vrams, "o-", label=f"KV {qt}")
                    has_data = True
            if has_data:
                ax.set_xlabel("Prompt Length (tokens)")
                ax.set_ylabel("Peak VRAM (MB)")
                ax.set_title(f"KV-Cache Memory Growth per Token{' — ' + model_name if suffix else ''}")
                ax.legend()
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, f"kv_memory_per_token{suffix}.png"))
            plt.close(fig)

    print(f"KV-cache plots saved to {output_dir}")


# ──────────────────────────────────────────────────────────────────────
# Final Benchmark Plots (Week 10)
# ──────────────────────────────────────────────────────────────────────

def plot_final_benchmark(output_dir):
    """Generate final consolidated benchmark plots."""
    week10_dir = os.path.join(RESULTS_DIR, "week10_full_benchmark")
    datasets, model_names = _load_from_week(week10_dir, "final_benchmark_table.json")

    if not datasets:
        print("No final benchmark data found. Skipping final plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for data, model_name in zip(datasets, model_names):
        if not isinstance(data, list):
            continue
        df = pd.DataFrame(data)
        suffix = f"_{model_name}" if model_name != "default" else ""

        df_bs1 = df[df["batch_size"] == 1] if "batch_size" in df.columns else df

        # Multi-config latency + throughput comparison
        if not df_bs1.empty:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            for config in df_bs1["config_id"].unique():
                subset = df_bs1[df_bs1["config_id"] == config].sort_values("prompt_length")
                axes[0].plot(
                    subset["prompt_length"], subset["latency_p95_ms"],
                    "o-", label=config,
                )
            axes[0].set_xlabel("Prompt Length (tokens)")
            axes[0].set_ylabel("p95 Latency (ms)")
            axes[0].set_title(f"All Configurations — p95 Latency{' — ' + model_name if suffix else ''}")
            axes[0].legend(fontsize=8)

            for config in df_bs1["config_id"].unique():
                subset = df_bs1[df_bs1["config_id"] == config].sort_values("prompt_length")
                axes[1].plot(
                    subset["prompt_length"], subset["throughput_tok_per_s"],
                    "o-", label=config,
                )
            axes[1].set_xlabel("Prompt Length (tokens)")
            axes[1].set_ylabel("Throughput (tokens/s)")
            axes[1].set_title(f"All Configurations — Throughput{' — ' + model_name if suffix else ''}")
            axes[1].legend(fontsize=8)

            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"final_latency_throughput{suffix}.png"))
            plt.close(fig)

        # VRAM scaling
        if not df_bs1.empty and "peak_vram_mb" in df_bs1.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            for config in df_bs1["config_id"].unique():
                subset = df_bs1[df_bs1["config_id"] == config].sort_values("prompt_length")
                ax.plot(subset["prompt_length"], subset["peak_vram_mb"], "o-", label=config)
            ax.set_xlabel("Prompt Length (tokens)")
            ax.set_ylabel("Peak VRAM (MB)")
            ax.set_title(f"All Configurations — VRAM Scaling{' — ' + model_name if suffix else ''}")
            ax.axhline(y=4096, color="gray", linestyle="--", alpha=0.5, label="4 GB limit")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"final_vram_scaling{suffix}.png"))
            plt.close(fig)

    # OOM threshold comparison
    oom_datasets, oom_models = _load_from_week(week10_dir, "oom_thresholds.json")
    if oom_datasets:
        for oom_data, model_name in zip(oom_datasets, oom_models):
            suffix = f"_{model_name}" if model_name != "default" else ""
            fig, ax = plt.subplots(figsize=(10, 5))
            configs = []
            thresholds = []
            for config, info in oom_data.items():
                threshold = info.get("oom_threshold_tokens", 0)
                if threshold > 0:
                    configs.append(config)
                    thresholds.append(threshold)

            if configs:
                ax.barh(configs, thresholds, color=sns.color_palette("colorblind"))
                ax.set_xlabel("OOM Threshold (tokens)")
                ax.set_title(f"Maximum Context Length Before OOM{' — ' + model_name if suffix else ''}")
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, f"oom_threshold_comparison{suffix}.png"))
            plt.close(fig)

    # KV-cache memory per token (also in week10_final for report convenience)
    week07_dir = os.path.join(RESULTS_DIR, "week07_kv_cache")
    mem_datasets, mem_models = _load_from_week(week07_dir, "kv_memory_per_token.json")
    if mem_datasets:
        for mem_data, model_name in zip(mem_datasets, mem_models):
            suffix = f"_{model_name}" if model_name != "default" else ""
            fig, ax = plt.subplots(figsize=(8, 5))
            kv_types = []
            mb_per_tok = []
            for qt, info in mem_data.items():
                est = info.get("estimated_mb_per_token")
                if est:
                    kv_types.append(f"KV {qt.upper()}")
                    mb_per_tok.append(est)
            if kv_types:
                bars = ax.bar(kv_types, mb_per_tok, color=sns.color_palette("colorblind"))
                ax.set_ylabel("MB per Token")
                ax.set_title(f"KV-Cache Memory per Token{' — ' + model_name if suffix else ''}")
                for bar, val in zip(bars, mb_per_tok):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=10)
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, f"kv_memory_per_token{suffix}.png"))
            plt.close(fig)

    # Benchmark accuracy
    week08_dir = os.path.join(RESULTS_DIR, "week08_kv_experiments")
    qual_datasets, qual_models = _load_from_week(week08_dir, "baseline_quality.json")
    if qual_datasets:
        for quality_data, model_name in zip(qual_datasets, qual_models):
            suffix = f"_{model_name}" if model_name != "default" else ""
            datasets_info = quality_data.get("datasets", {})
            if datasets_info:
                fig, ax = plt.subplots(figsize=(8, 5))
                names = []
                accuracies = []
                for ds_name, ds_data in datasets_info.items():
                    if "accuracy" in ds_data:
                        names.append(ds_name)
                        accuracies.append(ds_data["accuracy"] * 100)
                if names:
                    bars = ax.bar(names, accuracies, color=sns.color_palette("colorblind"))
                    ax.set_ylabel("Accuracy (%)")
                    ax.set_title(f"0-Shot Quality Evaluation{' — ' + model_name if suffix else ''}")
                    ax.set_ylim(0, 100)
                    for bar, acc in zip(bars, accuracies):
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10)
                    fig.tight_layout()
                    fig.savefig(os.path.join(output_dir, f"benchmark_accuracy{suffix}.png"))
                plt.close(fig)

    print(f"Final benchmark plots saved to {output_dir}")


# ──────────────────────────────────────────────────────────────────────
# Quality Plots (Week 08)
# ──────────────────────────────────────────────────────────────────────

def plot_quality_results(output_dir):
    """Plot quality evaluation results."""
    week08_dir = os.path.join(RESULTS_DIR, "week08_kv_experiments")
    datasets, model_names = _load_from_week(week08_dir, "baseline_quality.json")

    if not datasets:
        print("No quality data found. Skipping quality plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for data, model_name in zip(datasets, model_names):
        suffix = f"_{model_name}" if model_name != "default" else ""
        ds_info = data.get("datasets", {})
        if ds_info:
            fig, ax = plt.subplots(figsize=(8, 5))
            names = []
            accuracies = []
            for ds_name, ds_data in ds_info.items():
                if "accuracy" in ds_data:
                    names.append(ds_name)
                    accuracies.append(ds_data["accuracy"] * 100)

            if names:
                bars = ax.bar(names, accuracies, color=sns.color_palette("colorblind"))
                ax.set_ylabel("Accuracy (%)")
                ax.set_title(f"Quality Evaluation — 0-Shot Accuracy{' — ' + model_name if suffix else ''}")
                ax.set_ylim(0, 100)

                for bar, acc in zip(bars, accuracies):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{acc:.1f}%", ha="center", va="bottom", fontsize=10,
                    )

                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, f"quality_accuracy{suffix}.png"))
            plt.close(fig)

    print(f"Quality plots saved to {output_dir}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Generating All Plots")
    print("=" * 60)

    plots_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_baseline_scaling(os.path.join(plots_dir, "week03_baseline"))
    plot_attention_comparison(os.path.join(plots_dir, "week06_attention"))
    plot_kv_cache_tradeoffs(os.path.join(plots_dir, "week08_kv_cache"))
    plot_quality_results(os.path.join(plots_dir, "week08_quality"))
    plot_final_benchmark(os.path.join(plots_dir, "week10_final"))

    print("=" * 60)
    print(f"All plots saved to: {plots_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
