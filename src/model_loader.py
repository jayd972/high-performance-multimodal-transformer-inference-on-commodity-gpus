"""
Model loading with 4-bit weight quantization via bitsandbytes.

Provides functions to load transformer models within the VRAM budget
and report their memory footprint.
"""

import os
import sys
import logging
from typing import Tuple, Dict, Optional, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VRAM_WEIGHT_BUDGET_GB

logger = logging.getLogger("llm_inference.model_loader")


def get_bnb_4bit_config() -> BitsAndBytesConfig:
    """Standard 4-bit quantization config using NF4."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # nested quantization for extra savings
    )


def load_model_4bit(
    model_id: str,
    device_map: str = "auto",
    attn_implementation: Optional[str] = None,
    trust_remote_code: bool = True,
    max_memory: Optional[Dict] = None,
) -> Tuple[Any, Any]:
    """
    Load a model with 4-bit quantization.
    
    Args:
        model_id: HuggingFace model identifier.
        device_map: Device placement strategy.
        attn_implementation: Attention implementation override
            ("sdpa", "flash_attention_2", "eager", or None for default).
        trust_remote_code: Whether to trust remote code.
        max_memory: Optional max memory mapping, e.g. {0: "3.2GiB", "cpu": "16GiB"}.
    
    Returns:
        (model, tokenizer) tuple.
    """
    logger.info(f"Loading model: {model_id}")
    logger.info(f"Quantization: 4-bit NF4 with double quantization")

    quant_config = get_bnb_4bit_config()

    # Set max_memory to enforce VRAM budget if not explicitly provided
    if max_memory is None and torch.cuda.is_available():
        max_memory = {
            0: f"{VRAM_WEIGHT_BUDGET_GB}GiB",
            "cpu": "16GiB",
        }
        logger.info(f"VRAM budget: {VRAM_WEIGHT_BUDGET_GB} GiB for weights")

    # Tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Model
    logger.info("Loading model with 4-bit quantization...")
    model_kwargs = dict(
        quantization_config=quant_config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16,
    )
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    # Report footprint
    footprint = get_model_memory_footprint(model)
    logger.info(
        f"Model loaded — footprint: {footprint['total_mb']:.0f} MB "
        f"({footprint['total_gb']:.2f} GB)"
    )

    return model, tokenizer


def load_model_fp16(
    model_id: str,
    device_map: str = "auto",
    attn_implementation: Optional[str] = None,
    trust_remote_code: bool = True,
) -> Tuple[Any, Any]:
    """
    Load a model in FP16 (no quantization). Used for comparison / baseline
    only if VRAM permits.
    """
    logger.info(f"Loading model (FP16, no quantization): {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = dict(
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16,
    )
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    return model, tokenizer


def get_model_memory_footprint(model) -> Dict[str, float]:
    """
    Compute model weight memory footprint.
    
    Returns dict with total_bytes, total_mb, total_gb, and
    per-dtype breakdown.
    """
    total_bytes = 0
    dtype_bytes = {}

    for name, param in model.named_parameters():
        param_bytes = param.nelement() * param.element_size()
        total_bytes += param_bytes
        dtype_name = str(param.dtype)
        dtype_bytes[dtype_name] = dtype_bytes.get(dtype_name, 0) + param_bytes

    # Also count buffers
    for name, buf in model.named_buffers():
        buf_bytes = buf.nelement() * buf.element_size()
        total_bytes += buf_bytes

    return {
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 ** 2),
        "total_gb": total_bytes / (1024 ** 3),
        "dtype_breakdown_mb": {
            k: v / (1024 ** 2) for k, v in dtype_bytes.items()
        },
    }


def verify_vram_budget(model, budget_gb: float = VRAM_WEIGHT_BUDGET_GB) -> Dict:
    """
    Check whether the loaded model fits within the VRAM budget.
    
    Returns a dict with pass/fail and details.
    """
    footprint = get_model_memory_footprint(model)
    within_budget = footprint["total_gb"] <= budget_gb
    return {
        "within_budget": within_budget,
        "model_weight_gb": round(footprint["total_gb"], 3),
        "budget_gb": budget_gb,
        "headroom_gb": round(budget_gb - footprint["total_gb"], 3),
        "dtype_breakdown_mb": footprint["dtype_breakdown_mb"],
    }
