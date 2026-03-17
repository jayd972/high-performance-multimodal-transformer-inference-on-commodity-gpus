"""
Central configuration for High-Performance Multimodal Transformer Inference
on Commodity GPUs project.

All benchmark settings, model candidates, VRAM budgets, and evaluation
parameters are defined here and held constant across all experiments.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "prompts")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

# ──────────────────────────────────────────────────────────────────────
# Model candidates (2B–4B parameter range)
# ──────────────────────────────────────────────────────────────────────
MODEL_CANDIDATES: Dict[str, dict] = {
    "qwen2.5-3b": {
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "params_b": 3.0,
        "est_4bit_gb": 1.8,
        "notes": "Strong chat model, well-tested with bitsandbytes 4-bit.",
    },
    "phi-2": {
        "hf_id": "microsoft/phi-2",
        "params_b": 2.7,
        "est_4bit_gb": 1.5,
        "notes": "Good reasoning, MIT license, base model (no chat template).",
    },
    "gemma-2-2b": {
        "hf_id": "google/gemma-2-2b-it",
        "params_b": 2.0,
        "est_4bit_gb": 1.2,
        "notes": "Compact, instruction-tuned, good quality for size.",
    },
}

# Primary model selection
DEFAULT_MODEL_KEY = "qwen2.5-3b"
DEFAULT_MODEL_ID = MODEL_CANDIDATES[DEFAULT_MODEL_KEY]["hf_id"]

# ──────────────────────────────────────────────────────────────────────
# VRAM budget
# ──────────────────────────────────────────────────────────────────────
VRAM_TOTAL_GB = 4.0
VRAM_WEIGHT_BUDGET_GB = 3.2   # Max weight footprint
VRAM_HEADROOM_GB = VRAM_TOTAL_GB - VRAM_WEIGHT_BUDGET_GB  # ~0.8 GB


# ──────────────────────────────────────────────────────────────────────
# Benchmark settings (fixed across all experiments)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class BenchmarkConfig:
    """Fixed benchmark settings held constant across configurations."""
    prompt_lengths: List[int] = field(default_factory=lambda: [128, 512, 1024])
    output_limit: int = 64           # Reduced from 128 for faster runs on laptop GPU
    batch_sizes: List[int] = field(default_factory=lambda: [1])  # Batch=1 only to avoid OOM
    temperature: float = 0.0        # greedy decoding
    do_sample: bool = False
    top_p: float = 1.0
    num_warmup_runs: int = 2
    num_benchmark_runs: int = 5     # Reduced for faster iteration on laptop
    seed: int = 42


# ──────────────────────────────────────────────────────────────────────
# Correctness settings
# ──────────────────────────────────────────────────────────────────────
@dataclass
class CorrectnessConfig:
    """Correctness verification via deterministic greedy decoding."""
    num_logit_comparison_steps: int = 20
    max_logit_diff_tolerance: float = 0.01
    num_fixed_prompts: int = 5


# ──────────────────────────────────────────────────────────────────────
# Quality evaluation settings
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QualityEvalConfig:
    """0-shot quality evaluation on lightweight benchmarks."""
    datasets: List[str] = field(
        default_factory=lambda: ["ai2_arc", "google/boolq", "Rowan/hellaswag"]
    )
    dataset_display_names: List[str] = field(
        default_factory=lambda: ["ARC-Easy", "BoolQ", "HellaSwag"]
    )
    num_examples_per_dataset: int = 200
    max_runtime_per_config_minutes: int = 60


# ──────────────────────────────────────────────────────────────────────
# Workload types
# ──────────────────────────────────────────────────────────────────────
WORKLOAD_TYPES = [
    "single_turn",          # Single prompt → generation
    "multi_turn_short",     # 3-turn conversational exchange
    "long_context_stress",  # Maximum context length stress test
]

# ──────────────────────────────────────────────────────────────────────
# Attention backends to evaluate
# ──────────────────────────────────────────────────────────────────────
ATTENTION_BACKENDS = [
    "sdpa_flash",           # PyTorch SDPA with flash attention
    "sdpa_mem_efficient",   # PyTorch SDPA with memory-efficient attention
    "sdpa_math",            # PyTorch SDPA with math fallback
    "eager",                # Standard eager attention (no SDPA)
]

# ──────────────────────────────────────────────────────────────────────
# KV-cache quantization settings
# ──────────────────────────────────────────────────────────────────────
KV_CACHE_QUANT_TYPES = ["fp16", "int4", "int2"]

# ──────────────────────────────────────────────────────────────────────
# Configuration IDs for final benchmark matrix
# ──────────────────────────────────────────────────────────────────────
CONFIGURATION_IDS = {
    "baseline":          "FP16 KV-cache, eager attention",
    "sdpa_flash":        "FP16 KV-cache, SDPA flash attention",
    "sdpa_mem_eff":      "FP16 KV-cache, SDPA memory-efficient attention",
    "kv_int4":           "INT4 KV-cache, eager attention",
    "kv_int2":           "INT2 KV-cache, eager attention",
    "combined_flash_i4": "INT4 KV-cache, SDPA flash attention",
    "combined_memeff_i4":"INT4 KV-cache, SDPA memory-efficient attention",
}

# ──────────────────────────────────────────────────────────────────────
# Default instances
# ──────────────────────────────────────────────────────────────────────
BENCHMARK_CFG = BenchmarkConfig()
CORRECTNESS_CFG = CorrectnessConfig()
QUALITY_CFG = QualityEvalConfig()
