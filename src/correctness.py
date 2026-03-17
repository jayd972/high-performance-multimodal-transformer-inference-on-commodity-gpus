"""
Correctness verification module.

Verifies that optimisations do not change model output beyond a
small numerical tolerance by comparing:
  1. Top-1 token agreement rate across fixed prompts
  2. Maximum absolute logit difference for a fixed number of steps
"""

import os
import sys
import logging
from typing import Dict, List, Any, Tuple, Optional

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import set_seed

logger = logging.getLogger("llm_inference.correctness")


def collect_logits_trace(
    model,
    tokenizer,
    prompt: str,
    num_steps: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Collect token-by-token logits for a deterministic greedy trace.
    
    Args:
        model: The loaded model.
        tokenizer: The tokenizer.
        prompt: Input prompt.
        num_steps: Number of generation steps to trace.
        seed: Random seed for reproducibility.
    
    Returns:
        Dict with token_ids, top_k_tokens, and full logits for each step.
    """
    set_seed(seed)
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    trace = {
        "prompt": prompt,
        "prompt_tokens": input_ids.shape[-1],
        "num_steps": num_steps,
        "token_ids": [],
        "top1_tokens": [],
        "top5_tokens": [],
        "logits_at_step": [],  # Store top-k logits only to save memory
    }

    current_ids = input_ids.clone()

    for step in range(num_steps):
        with torch.no_grad():
            outputs = model(current_ids)
            next_token_logits = outputs.logits[:, -1, :]  # [1, vocab_size]

        # Top-1
        top1_id = next_token_logits.argmax(dim=-1).item()
        top1_token = tokenizer.decode([top1_id])

        # Top-5
        top5_values, top5_indices = torch.topk(next_token_logits, k=5, dim=-1)
        top5_info = [
            {
                "id": idx.item(),
                "token": tokenizer.decode([idx.item()]),
                "logit": val.item(),
            }
            for val, idx in zip(top5_values[0], top5_indices[0])
        ]

        trace["token_ids"].append(top1_id)
        trace["top1_tokens"].append(top1_token)
        trace["top5_tokens"].append(top5_info)
        trace["logits_at_step"].append(
            next_token_logits[0].float().cpu().numpy().tolist()
        )

        # Append token and continue
        current_ids = torch.cat(
            [current_ids, torch.tensor([[top1_id]], device=device)], dim=-1
        )

        # Stop if EOS
        if top1_id == tokenizer.eos_token_id:
            break

    trace["num_actual_steps"] = len(trace["token_ids"])
    return trace


def compare_traces(
    baseline_trace: Dict,
    test_trace: Dict,
) -> Dict[str, Any]:
    """
    Compare two logit traces and compute correctness metrics.
    
    Returns:
        Dict with top-1 agreement rate, max logit diff, etc.
    """
    n_steps = min(
        len(baseline_trace["token_ids"]),
        len(test_trace["token_ids"]),
    )

    if n_steps == 0:
        return {"error": "No steps to compare", "num_steps": 0}

    # Top-1 agreement
    agreements = 0
    for i in range(n_steps):
        if baseline_trace["token_ids"][i] == test_trace["token_ids"][i]:
            agreements += 1

    agreement_rate = agreements / n_steps

    # Max absolute logit difference
    max_logit_diffs = []
    mean_logit_diffs = []

    for i in range(n_steps):
        bl = np.array(baseline_trace["logits_at_step"][i])
        tl = np.array(test_trace["logits_at_step"][i])

        if bl.shape != tl.shape:
            logger.warning(f"Logit shape mismatch at step {i}: {bl.shape} vs {tl.shape}")
            min_len = min(len(bl), len(tl))
            bl = bl[:min_len]
            tl = tl[:min_len]

        abs_diff = np.abs(bl - tl)
        max_logit_diffs.append(float(np.max(abs_diff)))
        mean_logit_diffs.append(float(np.mean(abs_diff)))

    # First divergence point
    first_divergence = n_steps  # No divergence
    for i in range(n_steps):
        if baseline_trace["token_ids"][i] != test_trace["token_ids"][i]:
            first_divergence = i
            break

    return {
        "num_steps_compared": n_steps,
        "top1_agreement_rate": round(agreement_rate, 4),
        "top1_agreements": agreements,
        "top1_disagreements": n_steps - agreements,
        "first_divergence_step": first_divergence,
        "max_absolute_logit_diff": round(float(max(max_logit_diffs)), 6),
        "mean_max_logit_diff": round(float(np.mean(max_logit_diffs)), 6),
        "mean_mean_logit_diff": round(float(np.mean(mean_logit_diffs)), 6),
        "per_step_max_diffs": [round(d, 6) for d in max_logit_diffs],
    }


def run_correctness_suite(
    model,
    tokenizer,
    prompts: List[str],
    baseline_traces: Optional[List[Dict]] = None,
    num_steps: int = 50,
    config_id: str = "test",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run correctness verification on a set of prompts.
    
    If baseline_traces is None, this run IS the baseline (just collect traces).
    If baseline_traces is provided, compare against them.
    """
    logger.info(f"Running correctness suite: config={config_id}, {len(prompts)} prompts")

    traces = []
    for i, prompt in enumerate(prompts):
        logger.info(f"  Collecting trace {i+1}/{len(prompts)}...")
        trace = collect_logits_trace(
            model, tokenizer, prompt,
            num_steps=num_steps, seed=seed,
        )
        traces.append(trace)

    result = {
        "config_id": config_id,
        "num_prompts": len(prompts),
        "num_steps_per_prompt": num_steps,
        "traces": traces,
    }

    # Compare if baseline provided
    if baseline_traces is not None:
        comparisons = []
        for i in range(min(len(traces), len(baseline_traces))):
            comp = compare_traces(baseline_traces[i], traces[i])
            comp["prompt_index"] = i
            comparisons.append(comp)

        # Aggregate
        if comparisons:
            all_agreements = [c["top1_agreement_rate"] for c in comparisons]
            all_max_diffs = [c["max_absolute_logit_diff"] for c in comparisons]

            result["comparisons"] = comparisons
            result["aggregate"] = {
                "mean_agreement_rate": round(float(np.mean(all_agreements)), 4),
                "min_agreement_rate": round(float(min(all_agreements)), 4),
                "max_absolute_logit_diff": round(float(max(all_max_diffs)), 6),
                "mean_max_logit_diff": round(float(np.mean(all_max_diffs)), 6),
            }

            logger.info(
                f"  Correctness: agreement={result['aggregate']['mean_agreement_rate']:.2%}, "
                f"max_logit_diff={result['aggregate']['max_absolute_logit_diff']:.6f}"
            )

    return result
