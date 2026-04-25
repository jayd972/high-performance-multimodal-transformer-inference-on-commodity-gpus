"""
Core benchmarking harness for inference measurement.

Measures p50/p95 latency, tokens/s throughput, peak VRAM,
CPU RAM, and out-of-memory thresholds as a function of
sequence length.
"""

import os
import sys
import gc
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BENCHMARK_CFG
from src.vram_monitor import VRAMMonitor, reset_peak_memory, get_peak_memory_mb
from src.utils import Timer, set_seed

logger = logging.getLogger("llm_inference.benchmark")


@dataclass
class InferenceResult:
    """Result of a single inference run."""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_time_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    tokens_per_second: float = 0.0
    peak_vram_mb: float = 0.0
    peak_cpu_ram_mb: float = 0.0
    generated_text: str = ""
    oom_error: bool = False
    error_msg: str = ""


@dataclass
class BenchmarkResult:
    """Aggregated result over multiple inference runs."""
    config_id: str = ""
    prompt_length: int = 0
    output_limit: int = 0
    batch_size: int = 1
    num_runs: int = 0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0
    throughput_tokens_per_s: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    peak_vram_mb: float = 0.0
    peak_cpu_ram_mb: float = 0.0
    oom_occurred: bool = False
    individual_runs: List[Dict] = field(default_factory=list)


def prepare_prompt_at_length(
    tokenizer,
    base_text: str,
    target_length: int,
) -> str:
    """
    Truncate or pad a base text to exactly target_length tokens.
    
    Pads by repeating the text if necessary.
    """
    tokens = tokenizer.encode(base_text, add_special_tokens=False)

    if len(tokens) >= target_length:
        tokens = tokens[:target_length]
    else:
        # Repeat tokens to reach target length
        repeats_needed = (target_length // len(tokens)) + 1
        tokens = (tokens * repeats_needed)[:target_length]

    return tokenizer.decode(tokens, skip_special_tokens=True)


def run_single_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = BENCHMARK_CFG.output_limit,
    do_sample: bool = False,
    temperature: float = 0.0,
    monitor: Optional[VRAMMonitor] = None,
) -> InferenceResult:
    """
    Run a single inference pass and measure performance.
    """
    result = InferenceResult()
    device = next(model.parameters()).device

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        result.prompt_tokens = inputs["input_ids"].shape[-1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            reset_peak_memory()

        if monitor:
            monitor.start_polling()

        # Generation with timing
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        with Timer("generation") as timer:
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)

        if monitor:
            monitor.stop_polling()

        result.total_time_ms = timer.elapsed_ms
        result.generated_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]

        if result.generated_tokens > 0 and result.total_time_ms > 0:
            result.tokens_per_second = (
                result.generated_tokens / (result.total_time_ms / 1000)
            )

        # Memory stats
        if torch.cuda.is_available():
            result.peak_vram_mb = get_peak_memory_mb()
        if monitor:
            summary = monitor.get_summary()
            result.peak_cpu_ram_mb = summary.get("peak_cpu_ram_used_mb", 0)

        # Decode output
        result.generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

    except torch.cuda.OutOfMemoryError:
        result.oom_error = True
        result.error_msg = "CUDA Out of Memory"
        logger.warning(f"OOM at prompt_tokens={result.prompt_tokens}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        result.oom_error = False
        result.error_msg = str(e)
        logger.error(f"Inference error: {e}")

    return result


def run_batch_inference(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = BENCHMARK_CFG.output_limit,
    do_sample: bool = False,
    temperature: float = 0.0,
) -> InferenceResult:
    """Run batched inference."""
    result = InferenceResult()
    device = next(model.parameters()).device

    try:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        result.prompt_tokens = inputs["input_ids"].shape[-1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            reset_peak_memory()

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        with Timer("batch_generation") as timer:
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)

        result.total_time_ms = timer.elapsed_ms
        result.generated_tokens = (
            (outputs.shape[-1] - inputs["input_ids"].shape[-1]) * len(prompts)
        )

        if result.generated_tokens > 0 and result.total_time_ms > 0:
            result.tokens_per_second = (
                result.generated_tokens / (result.total_time_ms / 1000)
            )

        if torch.cuda.is_available():
            result.peak_vram_mb = get_peak_memory_mb()

    except torch.cuda.OutOfMemoryError:
        result.oom_error = True
        result.error_msg = "CUDA Out of Memory (batch)"
        logger.warning(f"OOM in batch inference, batch_size={len(prompts)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        result.error_msg = str(e)
        logger.error(f"Batch inference error: {e}")

    return result


def run_benchmark_sweep(
    model,
    tokenizer,
    base_prompt: str,
    config_id: str = "baseline",
    prompt_lengths: List[int] = None,
    output_limit: int = BENCHMARK_CFG.output_limit,
    batch_sizes: List[int] = None,
    num_warmup: int = BENCHMARK_CFG.num_warmup_runs,
    num_runs: int = BENCHMARK_CFG.num_benchmark_runs,
    seed: int = BENCHMARK_CFG.seed,
) -> List[BenchmarkResult]:
    """
    Run a full benchmark sweep across prompt lengths and batch sizes.
    
    Returns a list of BenchmarkResult objects.
    """
    if prompt_lengths is None:
        prompt_lengths = list(BENCHMARK_CFG.prompt_lengths)
    if batch_sizes is None:
        batch_sizes = list(BENCHMARK_CFG.batch_sizes)

    set_seed(seed)
    results = []
    monitor = VRAMMonitor()

    for prompt_len in prompt_lengths:
        prompt = prepare_prompt_at_length(tokenizer, base_prompt, prompt_len)

        for batch_size in batch_sizes:
            logger.info(
                f"Benchmarking config={config_id}, "
                f"prompt_len={prompt_len}, batch_size={batch_size}"
            )

            bench = BenchmarkResult(
                config_id=config_id,
                prompt_length=prompt_len,
                output_limit=output_limit,
                batch_size=batch_size,
            )

            # Warmup
            logger.info(f"  Warmup ({num_warmup} runs)...")
            for _ in range(num_warmup):
                if batch_size == 1:
                    run_single_inference(
                        model, tokenizer, prompt,
                        max_new_tokens=min(output_limit, 16),
                    )
                else:
                    run_batch_inference(
                        model, tokenizer, [prompt] * batch_size,
                        max_new_tokens=min(output_limit, 16),
                    )

            # Benchmark runs
            latencies = []
            throughputs = []
            peak_vrams = []
            peak_cpus = []
            oom_count = 0

            for run_idx in range(num_runs):
                monitor.reset()

                if batch_size == 1:
                    run_result = run_single_inference(
                        model, tokenizer, prompt,
                        max_new_tokens=output_limit,
                        monitor=monitor,
                    )
                else:
                    run_result = run_batch_inference(
                        model, tokenizer, [prompt] * batch_size,
                        max_new_tokens=output_limit,
                    )

                if run_result.oom_error:
                    oom_count += 1
                    bench.oom_occurred = True
                    break

                latencies.append(run_result.total_time_ms)
                throughputs.append(run_result.tokens_per_second)
                peak_vrams.append(run_result.peak_vram_mb)
                peak_cpus.append(run_result.peak_cpu_ram_mb)

                bench.individual_runs.append({
                    "run": run_idx,
                    "latency_ms": round(run_result.total_time_ms, 2),
                    "tokens_per_s": round(run_result.tokens_per_second, 2),
                    "peak_vram_mb": round(run_result.peak_vram_mb, 1),
                    "generated_tokens": run_result.generated_tokens,
                })

            if latencies:
                bench.num_runs = len(latencies)
                bench.latency_p50_ms = round(float(np.percentile(latencies, 50)), 2)
                bench.latency_p95_ms = round(float(np.percentile(latencies, 95)), 2)
                bench.latency_mean_ms = round(float(np.mean(latencies)), 2)
                bench.latency_std_ms = round(float(np.std(latencies)), 2)
                bench.throughput_tokens_per_s = round(float(np.mean(throughputs)), 2)
                bench.peak_vram_mb = round(float(max(peak_vrams)), 1)
                bench.peak_cpu_ram_mb = round(float(max(peak_cpus)), 1) if any(peak_cpus) else 0.0

            results.append(bench)
            logger.info(
                f"  -> p50={bench.latency_p50_ms}ms, p95={bench.latency_p95_ms}ms, "
                f"tok/s={bench.throughput_tokens_per_s}, "
                f"peak_vram={bench.peak_vram_mb}MB, OOM={bench.oom_occurred}"
            )

    return results


def find_oom_threshold(
    model,
    tokenizer,
    base_prompt: str,
    max_new_tokens: int = BENCHMARK_CFG.output_limit,
    start_length: int = 128,
    max_length: int = 8192,
    step: int = 128,
) -> Dict[str, Any]:
    """
    Linear scan for the maximum context length before OOM.
    
    Returns the threshold and measurements at each step.
    """
    logger.info(f"Searching for OOM threshold (start={start_length}, max={max_length})...")

    last_success = 0
    measurements = []

    length = start_length
    while length <= max_length:
        prompt = prepare_prompt_at_length(tokenizer, base_prompt, length)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        result = run_single_inference(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens,
        )

        measurement = {
            "prompt_length": length,
            "oom": result.oom_error,
            "latency_ms": round(result.total_time_ms, 2),
            "peak_vram_mb": round(result.peak_vram_mb, 1),
            "tokens_per_s": round(result.tokens_per_second, 2),
        }
        measurements.append(measurement)

        if result.oom_error:
            logger.info(f"  OOM at length={length}")
            break
        else:
            last_success = length
            logger.info(
                f"  OK at length={length}: {result.total_time_ms:.0f}ms, "
                f"VRAM={result.peak_vram_mb:.0f}MB"
            )
            length += step

    return {
        "oom_threshold_tokens": last_success,
        "max_tested_length": length,
        "measurements": measurements,
    }


def results_to_table(results: List[BenchmarkResult]) -> List[Dict]:
    """Convert benchmark results to flat table rows for CSV export."""
    rows = []
    for r in results:
        rows.append({
            "config_id": r.config_id,
            "prompt_length": r.prompt_length,
            "output_limit": r.output_limit,
            "batch_size": r.batch_size,
            "num_runs": r.num_runs,
            "latency_p50_ms": r.latency_p50_ms,
            "latency_p95_ms": r.latency_p95_ms,
            "latency_mean_ms": r.latency_mean_ms,
            "latency_std_ms": r.latency_std_ms,
            "throughput_tok_per_s": r.throughput_tokens_per_s,
            "peak_vram_mb": r.peak_vram_mb,
            "peak_cpu_ram_mb": r.peak_cpu_ram_mb,
            "oom_occurred": r.oom_occurred,
        })
    return rows
