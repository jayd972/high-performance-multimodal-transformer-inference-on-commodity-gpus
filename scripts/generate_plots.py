"""
Plot generation for all weeks.

Generates publication-quality plots from benchmark results:
  - Latency vs sequence length (per config)
  - VRAM vs sequence length (per config)
  - Throughput comparison bar charts
  - OOM threshold comparison
  - Quality retention comparison
  - KV-cache memory per token
"""

import os
import sys
import json

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


# ──────────────────────────────────────────────────────────────────────
# Baseline Plots (Week 03)
# ──────────────────────────────────────────────────────────────────────

def plot_baseline_scaling(output_dir):
    """Plot baseline latency and VRAM scaling with sequence length."""
    data = load_json(os.path.join(
        RESULTS_DIR, "week03_baseline", "baseline_benchmark_results.json"
    ))
    if not data:
        print("No baseline data found. Skipping baseline plots.")
        return

    df = pd.DataFrame(data)
    os.makedirs(output_dir, exist_ok=True)

    # Latency vs sequence length (batch_size=1)
    df_bs1 = df[df["batch_size"] == 1]
    if not df_bs1.empty:
        fig, ax = plt.subplots()
        ax.plot(df_bs1["prompt_length"], df_bs1["latency_p50_ms"], "o-", label="p50")
        ax.plot(df_bs1["prompt_length"], df_bs1["latency_p95_ms"], "s--", label="p95")
        ax.set_xlabel("Prompt Length (tokens)")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Baseline Latency vs Sequence Length (Batch=1)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "baseline_latency_vs_length.png"))
        plt.close(fig)

    # VRAM vs sequence length
    if not df_bs1.empty:
        fig, ax = plt.subplots()
        ax.plot(df_bs1["prompt_length"], df_bs1["peak_vram_mb"], "o-", color="red")
        ax.set_xlabel("Prompt Length (tokens)")
        ax.set_ylabel("Peak VRAM (MB)")
        ax.set_title("Baseline Peak VRAM vs Sequence Length (Batch=1)")
        ax.axhline(y=4096, color="gray", linestyle="--", alpha=0.5, label="4 GB limit")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "baseline_vram_vs_length.png"))
        plt.close(fig)

    print(f"Baseline plots saved to {output_dir}")


# ──────────────────────────────────────────────────────────────────────
# Attention Comparison Plots (Week 06)
# ──────────────────────────────────────────────────────────────────────

def plot_attention_comparison(output_dir):
    """Plot attention backend comparison."""
    data = load_json(os.path.join(
        RESULTS_DIR, "week06_attention_final", "attention_final_results.json"
    ))
    if not data:
        print("No attention data found. Skipping attention plots.")
        return

    df = pd.DataFrame(data)
    os.makedirs(output_dir, exist_ok=True)

    # Latency comparison grouped by prompt length (batch=1)
    df_bs1 = df[df["batch_size"] == 1]
    if not df_bs1.empty:
        fig, ax = plt.subplots()
        configs = df_bs1["config_id"].unique()
        x = np.arange(len(df_bs1["prompt_length"].unique()))
        width = 0.8 / len(configs)

        for i, config in enumerate(configs):
            subset = df_bs1[df_bs1["config_id"] == config].sort_values("prompt_length")
            ax.bar(
                x + i * width, subset["latency_p95_ms"],
                width, label=config, alpha=0.8,
            )

        ax.set_xlabel("Prompt Length (tokens)")
        ax.set_ylabel("p95 Latency (ms)")
        ax.set_title("Attention Backend Comparison — p95 Latency")
        ax.set_xticks(x + width * (len(configs) - 1) / 2)
        ax.set_xticklabels(sorted(df_bs1["prompt_length"].unique()))
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "attention_latency_comparison.png"))
        plt.close(fig)

    # Throughput comparison
    if not df_bs1.empty:
        fig, ax = plt.subplots()
        for config in df_bs1["config_id"].unique():
            subset = df_bs1[df_bs1["config_id"] == config].sort_values("prompt_length")
            ax.plot(
                subset["prompt_length"], subset["throughput_tok_per_s"],
                "o-", label=config,
            )
        ax.set_xlabel("Prompt Length (tokens)")
        ax.set_ylabel("Throughput (tokens/s)")
        ax.set_title("Attention Backend Comparison — Throughput")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "attention_throughput_comparison.png"))
        plt.close(fig)

    # VRAM comparison
    if not df_bs1.empty:
        fig, ax = plt.subplots()
        for config in df_bs1["config_id"].unique():
            subset = df_bs1[df_bs1["config_id"] == config].sort_values("prompt_length")
            ax.plot(
                subset["prompt_length"], subset["peak_vram_mb"],
                "o-", label=config,
            )
        ax.set_xlabel("Prompt Length (tokens)")
        ax.set_ylabel("Peak VRAM (MB)")
        ax.set_title("Attention Backend Comparison — Peak VRAM")
        ax.axhline(y=4096, color="gray", linestyle="--", alpha=0.5, label="4 GB limit")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "attention_vram_comparison.png"))
        plt.close(fig)

    print(f"Attention comparison plots saved to {output_dir}")


# ──────────────────────────────────────────────────────────────────────
# KV-Cache Plots (Week 08)
# ──────────────────────────────────────────────────────────────────────

def plot_kv_cache_tradeoffs(output_dir):
    """Plot KV-cache tradeoff curves."""
    data = load_json(os.path.join(
        RESULTS_DIR, "week08_kv_experiments", "kv_tradeoff_analysis.json"
    ))
    if not data:
        print("No KV tradeoff data found. Skipping KV plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    entries = data.get("per_prompt_length", {})
    if entries:
        # VRAM savings vs latency change
        fig, ax = plt.subplots()
        for key, val in entries.items():
            ax.scatter(
                val["vram_saved_pct"], val["latency_change_pct"],
                s=100, label=key, zorder=5,
            )
        ax.set_xlabel("VRAM Saved (%)")
        ax.set_ylabel("Latency Change (%)")
        ax.set_title("KV-Cache Quantization: Memory vs Latency Tradeoff")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "kv_vram_vs_latency_tradeoff.png"))
        plt.close(fig)

    # Memory per token
    mem_data = load_json(os.path.join(
        RESULTS_DIR, "week07_kv_cache", "kv_memory_per_token.json"
    ))
    if mem_data:
        fig, ax = plt.subplots()
        for qt, info in mem_data.items():
            measurements = info.get("measurements", [])
            valid = [m for m in measurements if not m.get("oom") and m["peak_vram_mb"] > 0]
            if valid:
                lengths = [m["prompt_length"] for m in valid]
                vrams = [m["peak_vram_mb"] for m in valid]
                ax.plot(lengths, vrams, "o-", label=f"KV {qt}")
        ax.set_xlabel("Prompt Length (tokens)")
        ax.set_ylabel("Peak VRAM (MB)")
        ax.set_title("KV-Cache Memory Growth per Token")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "kv_memory_per_token.png"))
        plt.close(fig)

    print(f"KV-cache plots saved to {output_dir}")


# ──────────────────────────────────────────────────────────────────────
# Final Benchmark Plots (Week 10)
# ──────────────────────────────────────────────────────────────────────

def plot_final_benchmark(output_dir):
    """Generate final consolidated benchmark plots."""
    data = load_json(os.path.join(
        RESULTS_DIR, "week10_full_benchmark", "final_benchmark_table.json"
    ))
    if not data:
        print("No final benchmark data found. Skipping final plots.")
        return

    df = pd.DataFrame(data)
    os.makedirs(output_dir, exist_ok=True)

    df_bs1 = df[df["batch_size"] == 1]

    # Multi-config latency comparison
    if not df_bs1.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # p95 Latency
        for config in df_bs1["config_id"].unique():
            subset = df_bs1[df_bs1["config_id"] == config].sort_values("prompt_length")
            axes[0].plot(
                subset["prompt_length"], subset["latency_p95_ms"],
                "o-", label=config,
            )
        axes[0].set_xlabel("Prompt Length (tokens)")
        axes[0].set_ylabel("p95 Latency (ms)")
        axes[0].set_title("All Configurations — p95 Latency")
        axes[0].legend(fontsize=8)

        # Throughput
        for config in df_bs1["config_id"].unique():
            subset = df_bs1[df_bs1["config_id"] == config].sort_values("prompt_length")
            axes[1].plot(
                subset["prompt_length"], subset["throughput_tok_per_s"],
                "o-", label=config,
            )
        axes[1].set_xlabel("Prompt Length (tokens)")
        axes[1].set_ylabel("Throughput (tokens/s)")
        axes[1].set_title("All Configurations — Throughput")
        axes[1].legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "final_latency_throughput.png"))
        plt.close(fig)

    # VRAM scaling across prompt lengths
    if not df_bs1.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for config in df_bs1["config_id"].unique():
            subset = df_bs1[df_bs1["config_id"] == config].sort_values("prompt_length")
            ax.plot(subset["prompt_length"], subset["peak_vram_mb"], "o-", label=config)
        ax.set_xlabel("Prompt Length (tokens)")
        ax.set_ylabel("Peak VRAM (MB)")
        ax.set_title("All Configurations — VRAM Scaling")
        ax.axhline(y=4096, color="gray", linestyle="--", alpha=0.5, label="4 GB limit")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "final_vram_scaling.png"))
        plt.close(fig)

    # OOM threshold comparison
    oom_data = load_json(os.path.join(
        RESULTS_DIR, "week10_full_benchmark", "oom_thresholds.json"
    ))
    if oom_data:
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
            ax.set_title("Maximum Context Length Before OOM")
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "oom_threshold_comparison.png"))
        plt.close(fig)

    # VRAM scaling comparison — FlashAttention-2 vs baseline from OOM data
    if oom_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        palette = sns.color_palette("colorblind")
        color_idx = 0
        for config in ["baseline", "flash_attention_2", "sdpa_default", "kv_int4"]:
            info = oom_data.get(config)
            if not info:
                continue
            measurements = info.get("measurements", [])
            valid = [m for m in measurements if not m.get("oom")]
            if valid:
                lengths = [m["prompt_length"] for m in valid]
                vrams = [m["peak_vram_mb"] for m in valid]
                ax.plot(lengths, vrams, "o-", label=config, color=palette[color_idx])
            color_idx += 1
        ax.set_xlabel("Context Length (tokens)")
        ax.set_ylabel("Peak VRAM (MB)")
        ax.set_title("VRAM Scaling — OOM Threshold Test")
        ax.axhline(y=4096, color="gray", linestyle="--", alpha=0.5, label="4 GB limit")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "vram_scaling_comparison.png"))
        plt.close(fig)

    # KV-cache memory per token (also in week10_final for report convenience)
    mem_data = load_json(os.path.join(
        RESULTS_DIR, "week07_kv_cache", "kv_memory_per_token.json"
    ))
    if mem_data:
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
            ax.set_title("KV-Cache Memory per Token by Precision")
            for bar, val in zip(bars, mb_per_tok):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "kv_memory_per_token.png"))
        plt.close(fig)

    # Benchmark accuracy (also in week10_final for report convenience)
    quality_data = load_json(os.path.join(
        RESULTS_DIR, "week08_kv_experiments", "baseline_quality.json"
    ))
    if quality_data:
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
                ax.set_title("0-Shot Quality Evaluation — Benchmark Accuracy")
                ax.set_ylim(0, 100)
                for bar, acc in zip(bars, accuracies):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            f"{acc:.1f}%", ha="center", va="bottom", fontsize=10)
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, "benchmark_accuracy.png"))
            plt.close(fig)

    print(f"Final benchmark plots saved to {output_dir}")


# ──────────────────────────────────────────────────────────────────────
# Quality Plots (Week 08)
# ──────────────────────────────────────────────────────────────────────

def plot_quality_results(output_dir):
    """Plot quality evaluation results."""
    data = load_json(os.path.join(
        RESULTS_DIR, "week08_kv_experiments", "baseline_quality.json"
    ))
    if not data:
        print("No quality data found. Skipping quality plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    datasets = data.get("datasets", {})
    if datasets:
        fig, ax = plt.subplots(figsize=(8, 5))
        names = []
        accuracies = []
        for ds_name, ds_data in datasets.items():
            if "accuracy" in ds_data:
                names.append(ds_name)
                accuracies.append(ds_data["accuracy"] * 100)

        if names:
            bars = ax.bar(names, accuracies, color=sns.color_palette("colorblind"))
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Quality Evaluation — 0-Shot Accuracy")
            ax.set_ylim(0, 100)

            # Add value labels
            for bar, acc in zip(bars, accuracies):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{acc:.1f}%", ha="center", va="bottom", fontsize=10,
                )

            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "quality_accuracy.png"))
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
