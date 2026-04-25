"""
Multimodal model loading and inference for vision-language models.

Supports Phi-3.5-vision-instruct and LLaVA-1.5-7B with:
  - 4-bit weight quantization via bitsandbytes
  - FlashAttention-2 (explicit, not just SDPA fallback)
  - KV-cache quantization via optimum-quanto
  - Combined FA2 + KV-cache configurations
"""

import os
import sys
import gc
import io
import logging
import time
from typing import Dict, List, Any, Optional, Tuple

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    VRAM_WEIGHT_BUDGET_GB, BENCHMARK_CFG, MODEL_CANDIDATES,
    MULTIMODAL_TEST_IMAGE_URL,
)
from src.model_loader import get_bnb_4bit_config, get_model_memory_footprint
from src.vram_monitor import VRAMMonitor, reset_peak_memory, get_peak_memory_mb
from src.utils import Timer

logger = logging.getLogger("llm_inference.multimodal")


def _check_flash_attn_available() -> bool:
    """Check whether the flash-attn package is importable."""
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def download_test_image(url: str = MULTIMODAL_TEST_IMAGE_URL):
    """Download and cache a test image for benchmarking."""
    from PIL import Image
    import urllib.request

    cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", ".image_cache",
    )
    os.makedirs(cache_dir, exist_ok=True)

    filename = url.split("/")[-1]
    cache_path = os.path.join(cache_dir, filename)

    if os.path.exists(cache_path):
        return Image.open(cache_path).convert("RGB")

    logger.info(f"Downloading test image from {url}")
    urllib.request.urlretrieve(url, cache_path)
    return Image.open(cache_path).convert("RGB")


def load_multimodal_model_4bit(
    model_key: str,
    attn_implementation: Optional[str] = None,
    max_memory: Optional[Dict] = None,
) -> Tuple[Any, Any]:
    """
    Load a multimodal model with 4-bit quantization.

    Args:
        model_key: Key in MODEL_CANDIDATES (e.g. "phi-3.5-vision", "llava-1.5-7b").
        attn_implementation: "eager", "sdpa", or "flash_attention_2".
        max_memory: Optional max memory mapping.

    Returns:
        (model, processor) tuple.
    """
    model_info = MODEL_CANDIDATES[model_key]
    model_id = model_info["hf_id"]
    trust_remote = model_info.get("trust_remote_code", True)

    logger.info(f"Loading multimodal model: {model_id}")
    logger.info(f"Quantization: 4-bit NF4 with double quantization")

    if attn_implementation == "flash_attention_2" and not _check_flash_attn_available():
        logger.warning("flash-attn not installed; falling back to sdpa")
        attn_implementation = "sdpa"

    quant_config = get_bnb_4bit_config()

    if max_memory is None and torch.cuda.is_available():
        max_memory = {
            0: f"{VRAM_WEIGHT_BUDGET_GB}GiB",
            "cpu": "16GiB",
        }

    if model_key == "llava-1.5-7b":
        return _load_llava(model_id, quant_config, attn_implementation,
                           trust_remote, max_memory)
    elif model_key == "phi-3.5-vision":
        return _load_phi_vision(model_id, quant_config, attn_implementation,
                                trust_remote, max_memory)
    else:
        raise ValueError(f"Unknown multimodal model key: {model_key}")


def _load_llava(model_id, quant_config, attn_implementation, trust_remote, max_memory):
    """Load LLaVA-1.5-7B via LlavaForConditionalGeneration."""
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=trust_remote,
    )

    model_kwargs = dict(
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=trust_remote,
        torch_dtype=torch.float16,
    )
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    model = LlavaForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    model.eval()

    footprint = get_model_memory_footprint(model)
    logger.info(
        f"LLaVA loaded — footprint: {footprint['total_mb']:.0f} MB "
        f"({footprint['total_gb']:.2f} GB)"
    )
    return model, processor


def _load_phi_vision(model_id, quant_config, attn_implementation, trust_remote, max_memory):
    """Load Phi-3.5-vision-instruct via AutoModelForCausalLM."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=trust_remote,
    )

    model_kwargs = dict(
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=trust_remote,
        torch_dtype=torch.float16,
    )
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    footprint = get_model_memory_footprint(model)
    logger.info(
        f"Phi-3.5-vision loaded — footprint: {footprint['total_mb']:.0f} MB "
        f"({footprint['total_gb']:.2f} GB)"
    )
    return model, processor


def prepare_multimodal_input(
    processor,
    model_key: str,
    prompt: str,
    image=None,
) -> Dict[str, torch.Tensor]:
    """
    Prepare inputs for a multimodal model.

    Args:
        processor: The model's processor (handles both text and image).
        model_key: Key in MODEL_CANDIDATES.
        prompt: Text prompt.
        image: PIL Image or None (downloads test image if None).

    Returns:
        Dict of tensors ready for model.generate().
    """
    if image is None:
        image = download_test_image()

    if model_key == "llava-1.5-7b":
        conversation = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]},
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=text, return_tensors="pt")
    elif model_key == "phi-3.5-vision":
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{prompt}"},
        ]
        text = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(text, images=[image], return_tensors="pt")
    else:
        raise ValueError(f"Unknown multimodal model key: {model_key}")

    return inputs


def run_multimodal_inference(
    model,
    processor,
    model_key: str,
    prompt: str,
    image=None,
    max_new_tokens: int = BENCHMARK_CFG.output_limit,
    do_sample: bool = False,
    kv_cache=None,
) -> Dict[str, Any]:
    """
    Run inference on a multimodal model and measure performance.

    Args:
        model: The loaded multimodal model.
        processor: The model's processor.
        model_key: Key in MODEL_CANDIDATES.
        prompt: Text prompt.
        image: PIL Image or None.
        max_new_tokens: Max tokens to generate.
        do_sample: Whether to sample.
        kv_cache: Optional QuantizedCache for KV-cache quantization.

    Returns:
        Dict with latency, throughput, VRAM, and generated text.
    """
    result = {
        "model_key": model_key,
        "status": "failed",
    }

    try:
        inputs = prepare_multimodal_input(processor, model_key, prompt, image)

        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        prompt_tokens = inputs.get("input_ids", torch.tensor([])).shape[-1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            reset_peak_memory()

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        if hasattr(processor, "tokenizer"):
            pad_id = getattr(processor.tokenizer, "pad_token_id", None)
            if pad_id is not None:
                gen_kwargs["pad_token_id"] = pad_id
        if kv_cache is not None:
            gen_kwargs["past_key_values"] = kv_cache

        with Timer("multimodal_gen") as timer:
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)

        generated_tokens = outputs.shape[-1] - prompt_tokens
        total_time_ms = timer.elapsed_ms

        if torch.cuda.is_available():
            peak_vram_mb = get_peak_memory_mb()
        else:
            peak_vram_mb = 0.0

        if hasattr(processor, "decode"):
            generated_text = processor.decode(
                outputs[0][prompt_tokens:], skip_special_tokens=True,
            )
        elif hasattr(processor, "tokenizer"):
            generated_text = processor.tokenizer.decode(
                outputs[0][prompt_tokens:], skip_special_tokens=True,
            )
        else:
            generated_text = ""

        result.update({
            "status": "success",
            "prompt_tokens": int(prompt_tokens),
            "generated_tokens": int(generated_tokens),
            "total_time_ms": round(total_time_ms, 2),
            "tokens_per_second": round(
                generated_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0, 2,
            ),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "generated_text": generated_text[:200],
        })

    except torch.cuda.OutOfMemoryError:
        result["error"] = "CUDA Out of Memory"
        result["oom_error"] = True
        logger.warning("OOM during multimodal inference")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Multimodal inference failed: {e}")

    return result


def benchmark_multimodal_model(
    model,
    processor,
    model_key: str,
    prompts: List[str],
    image=None,
    config_id: str = "baseline",
    attn_implementation: str = "eager",
    kv_quant_type: Optional[str] = None,
    max_new_tokens: int = BENCHMARK_CFG.output_limit,
    num_warmup: int = BENCHMARK_CFG.num_warmup_runs,
    num_runs: int = BENCHMARK_CFG.num_benchmark_runs,
) -> Dict[str, Any]:
    """
    Benchmark a multimodal model across prompts with optional KV-cache quantization.

    Returns aggregated performance metrics.
    """
    from src.kv_cache_quant import create_quantized_kv_cache

    logger.info(f"Benchmarking {model_key} config={config_id}")

    if image is None:
        image = download_test_image()

    kv_cache = None
    if kv_quant_type and kv_quant_type != "fp16":
        kv_cache = create_quantized_kv_cache(model.config, kv_quant_type)
        if kv_cache is None:
            logger.warning(f"KV-cache {kv_quant_type} not available, using fp16")

    # Warmup
    logger.info(f"  Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        run_multimodal_inference(
            model, processor, model_key, prompts[0],
            image=image, max_new_tokens=min(max_new_tokens, 16),
        )

    latencies = []
    throughputs = []
    peak_vrams = []
    errors = []

    for run_idx in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            reset_peak_memory()

        prompt = prompts[run_idx % len(prompts)]

        try:
            res = run_multimodal_inference(
                model, processor, model_key, prompt,
                image=image, max_new_tokens=max_new_tokens,
                kv_cache=kv_cache,
            )

            if res.get("oom_error"):
                errors.append(f"OOM on run {run_idx}")
                break

            if res["status"] != "success":
                errors.append(res.get("error", "Unknown error"))
                break

            latencies.append(res["total_time_ms"])
            throughputs.append(res["tokens_per_second"])
            peak_vrams.append(res["peak_vram_mb"])

        except Exception as e:
            errors.append(str(e))
            logger.warning(f"Run {run_idx} failed: {e}")
            break

    if latencies:
        result = {
            "config_id": config_id,
            "model_key": model_key,
            "attn_implementation": attn_implementation,
            "kv_quant_type": kv_quant_type or "fp16",
            "num_runs": len(latencies),
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "latency_mean_ms": round(float(np.mean(latencies)), 2),
            "throughput_tok_per_s": round(float(np.mean(throughputs)), 2),
            "peak_vram_mb": round(float(max(peak_vrams)), 1),
            "errors": errors,
            "status": "success",
        }
    else:
        result = {
            "config_id": config_id,
            "model_key": model_key,
            "status": "failed",
            "errors": errors,
        }

    logger.info(f"  -> {config_id}: {result.get('status')}")
    return result
