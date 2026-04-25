"""
KV-cache quantization experiments.

Evaluates the memory savings and quality impact of quantizing
the key-value cache during autoregressive generation.
Supports INT4 and INT2 KV-cache quantization via optimum-quanto.

Note: quanto only supports 2-bit and 4-bit; INT8 is NOT available.
"""

import os
import sys
import gc
import logging
import time
from typing import Dict, List, Any, Optional

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BENCHMARK_CFG, KV_CACHE_QUANT_TYPES
from src.vram_monitor import VRAMMonitor, reset_peak_memory, get_peak_memory_mb
from src.utils import Timer

logger = logging.getLogger("llm_inference.kv_cache")


# ──────────────────────────────────────────────────────────────────────
# KV-cache quantization via transformers QuantizedCache
# ──────────────────────────────────────────────────────────────────────

def check_kv_quant_support() -> Dict[str, bool]:
    """Check which KV-cache quantization methods are available."""
    support = {
        "transformers_kv_quant": False,
        "quanto_available": False,
        "supported_nbits": [],
    }

    try:
        from transformers.cache_utils import QuantizedCache
        support["transformers_kv_quant"] = True
    except ImportError:
        pass

    try:
        from optimum.quanto import qint4
        support["quanto_available"] = True
        support["supported_nbits"] = [2, 4]
    except ImportError:
        try:
            import quanto
            support["quanto_available"] = True
            support["supported_nbits"] = [2, 4]
        except ImportError:
            pass

    return support


def create_quantized_kv_cache(model_config, quant_type: str = "int4"):
    """
    Create a quantized KV-cache instance using transformers QuantizedCache.
    
    Quanto backend supports 2-bit and 4-bit quantization only.
    
    Args:
        model_config: The model's PretrainedConfig (model.config).
        quant_type: "int4" or "int2".
    
    Returns:
        QuantizedCache instance or None if not supported.
    """
    try:
        from transformers.cache_utils import QuantizedCache

        nbits_map = {"int2": 2, "int4": 4}
        nbits = nbits_map.get(quant_type)
        if nbits is None:
            logger.warning(f"Unsupported KV quant type '{quant_type}'. Quanto supports int2/int4 only.")
            return None

        cache = QuantizedCache(
            backend="quanto",
            config=model_config,
            nbits=nbits,
        )
        logger.info(f"Created QuantizedCache (quanto): {quant_type} ({nbits}-bit)")
        return cache

    except ImportError as e:
        logger.warning(f"KV-cache quantization not available: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error creating KV-cache: {e}")
        return None


def run_inference_with_kv_quant(
    model,
    tokenizer,
    prompt: str,
    kv_quant_type: str = "int4",
    max_new_tokens: int = BENCHMARK_CFG.output_limit,
    do_sample: bool = False,
) -> Dict[str, Any]:
    """
    Run inference with quantized KV-cache and measure memory.
    
    Args:
        kv_quant_type: "int4" or "int2" (quanto does NOT support int8).
    
    Returns performance metrics and the generated text.
    """
    device = next(model.parameters()).device
    result = {
        "kv_quant_type": kv_quant_type,
        "status": "failed",
    }

    try:
        kv_cache = create_quantized_kv_cache(model.config, kv_quant_type)
        if kv_cache is None:
            result["error"] = "KV-cache quantization not supported"
            return result

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_tokens = inputs["input_ids"].shape[-1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            reset_peak_memory()

        with Timer("kv_quant_gen") as timer:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    past_key_values=kv_cache,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                )

        generated_tokens = outputs.shape[-1] - prompt_tokens
        total_time_ms = timer.elapsed_ms

        if torch.cuda.is_available():
            peak_vram_mb = get_peak_memory_mb()
        else:
            peak_vram_mb = 0.0

        generated_text = tokenizer.decode(
            outputs[0][prompt_tokens:], skip_special_tokens=True
        )

        result.update({
            "status": "success",
            "prompt_tokens": prompt_tokens,
            "generated_tokens": int(generated_tokens),
            "total_time_ms": round(total_time_ms, 2),
            "tokens_per_second": round(
                generated_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0, 2
            ),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "generated_text": generated_text[:200],
        })

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"KV-cache quant inference failed: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return result


def measure_kv_memory_per_token(
    model,
    tokenizer,
    base_prompt: str,
    kv_quant_type: str = "fp16",
    lengths: List[int] = None,
    max_new_tokens: int = BENCHMARK_CFG.output_limit,
) -> Dict[str, Any]:
    """
    Measure VRAM growth as context length increases to estimate
    KV-cache memory per token.
    
    Args:
        kv_quant_type: "fp16" (no quant), "int4", or "int2".
    
    Returns:
        Dict with per-token memory estimate and measurements.
    """
    from src.benchmark_harness import prepare_prompt_at_length

    if lengths is None:
        lengths = [64, 128, 256, 512, 768, 1024]

    measurements = []

    for length in lengths:
        prompt = prepare_prompt_at_length(tokenizer, base_prompt, length)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            reset_peak_memory()

        try:
            if kv_quant_type == "fp16":
                # Standard inference (no KV quant)
                from src.benchmark_harness import run_single_inference
                res = run_single_inference(
                    model, tokenizer, prompt,
                    max_new_tokens=max_new_tokens,
                )
                measurements.append({
                    "prompt_length": length,
                    "peak_vram_mb": round(res.peak_vram_mb, 1),
                    "oom": res.oom_error,
                })
            else:
                res = run_inference_with_kv_quant(
                    model, tokenizer, prompt,
                    kv_quant_type=kv_quant_type,
                    max_new_tokens=max_new_tokens,
                )
                measurements.append({
                    "prompt_length": length,
                    "peak_vram_mb": res.get("peak_vram_mb", 0),
                    "oom": res.get("status") == "failed" and "OOM" in res.get("error", ""),
                })

        except Exception as e:
            measurements.append({
                "prompt_length": length,
                "peak_vram_mb": 0,
                "oom": True,
                "error": str(e),
            })
            break

    # Estimate memory per token via linear regression
    valid = [m for m in measurements if not m.get("oom", False) and m["peak_vram_mb"] > 0]
    per_token_mb = 0.0

    if len(valid) >= 2:
        x = np.array([m["prompt_length"] for m in valid])
        y = np.array([m["peak_vram_mb"] for m in valid])
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            per_token_mb = round(float(slope), 4)

    return {
        "kv_quant_type": kv_quant_type,
        "measurements": measurements,
        "estimated_mb_per_token": per_token_mb,
        "num_valid_measurements": len(valid),
    }


def compare_kv_quant_configs(
    model,
    tokenizer,
    prompt: str,
    quant_types: List[str] = None,
    max_new_tokens: int = BENCHMARK_CFG.output_limit,
    num_runs: int = 5,
) -> Dict[str, Dict]:
    """
    Compare FP16, INT4, and INT2 KV-cache configurations.
    
    Returns a dict mapping quant_type to aggregated metrics.
    Note: quanto only supports int2/int4. INT8 is NOT available.
    """
    from src.benchmark_harness import run_single_inference

    if quant_types is None:
        quant_types = list(KV_CACHE_QUANT_TYPES)  # ["fp16", "int4", "int2"]

    results = {}

    for qt in quant_types:
        logger.info(f"Benchmarking KV-cache type: {qt}")

        latencies = []
        vrams = []
        errors = []

        for i in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            try:
                if qt == "fp16":
                    res = run_single_inference(
                        model, tokenizer, prompt,
                        max_new_tokens=max_new_tokens,
                    )
                    if not res.oom_error:
                        latencies.append(res.total_time_ms)
                        vrams.append(res.peak_vram_mb)
                    else:
                        errors.append("OOM")
                        break
                else:
                    res = run_inference_with_kv_quant(
                        model, tokenizer, prompt,
                        kv_quant_type=qt,
                        max_new_tokens=max_new_tokens,
                    )
                    if res["status"] == "success":
                        latencies.append(res["total_time_ms"])
                        vrams.append(res["peak_vram_mb"])
                    else:
                        errors.append(res.get("error", "Unknown error"))
                        break

            except Exception as e:
                errors.append(str(e))
                break

        if latencies:
            results[qt] = {
                "num_runs": len(latencies),
                "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
                "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
                "latency_mean_ms": round(float(np.mean(latencies)), 2),
                "peak_vram_mb": round(float(max(vrams)), 1),
                "mean_vram_mb": round(float(np.mean(vrams)), 1),
                "status": "success",
                "errors": errors,
            }
        else:
            results[qt] = {
                "status": "failed",
                "errors": errors,
            }

        logger.info(f"  -> {qt}: {results[qt]}")

    return results
