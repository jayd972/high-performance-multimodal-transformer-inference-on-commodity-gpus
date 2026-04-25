"""
Quality evaluation on lightweight benchmarks.

Runs 0-shot evaluation on ARC-Easy, BoolQ, and HellaSwag using
log-likelihood scoring. Reports accuracy and quality retention
relative to baseline.
"""

import os
import sys
import gc
import time
import logging
from typing import Dict, List, Any, Optional

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import QUALITY_CFG

logger = logging.getLogger("llm_inference.quality")


def load_eval_dataset(
    dataset_name: str,
    num_examples: int = 200,
    seed: int = 42,
) -> List[Dict]:
    """
    Load evaluation examples from HuggingFace datasets.
    
    Supports: ai2_arc (ARC-Easy), google/boolq (BoolQ), Rowan/hellaswag.
    Returns a list of dicts with 'question', 'choices', 'answer_idx'.
    """
    from datasets import load_dataset

    examples = []

    if dataset_name == "ai2_arc":
        ds = load_dataset("ai2_arc", "ARC-Easy", split="test")
        ds = ds.shuffle(seed=seed).select(range(min(num_examples, len(ds))))

        for item in ds:
            choices = item["choices"]["text"]
            answer_key = item["answerKey"]
            if answer_key.isalpha():
                answer_idx = ord(answer_key.upper()) - ord("A")
            else:
                answer_idx = int(answer_key) - 1
            examples.append({
                "question": item["question"],
                "choices": choices,
                "answer_idx": answer_idx,
            })

    elif dataset_name == "google/boolq":
        ds = load_dataset("google/boolq", split="validation")
        ds = ds.shuffle(seed=seed).select(range(min(num_examples, len(ds))))

        for item in ds:
            examples.append({
                "question": item["passage"] + "\n\nQuestion: " + item["question"],
                "choices": ["No", "Yes"],
                "answer_idx": 1 if item["answer"] else 0,
            })

    elif dataset_name in ("Rowan/hellaswag", "hellaswag"):
        ds = load_dataset("Rowan/hellaswag", split="validation")
        ds = ds.shuffle(seed=seed).select(range(min(num_examples, len(ds))))

        for item in ds:
            examples.append({
                "question": item["ctx"],
                "choices": item["endings"],
                "answer_idx": int(item["label"]),
            })

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    logger.info(f"Loaded {len(examples)} examples from {dataset_name}")
    return examples


def score_choice_loglikelihood(
    model,
    tokenizer,
    context: str,
    choice: str,
) -> float:
    """
    Compute the log-likelihood of a choice given a context.
    
    Uses teacher forcing: encode context+choice, compute log-probs
    of the choice tokens conditioned on context.
    """
    device = next(model.parameters()).device

    # Encode context and full sequence
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    full_text = context + " " + choice
    full_ids = tokenizer.encode(full_text, add_special_tokens=True)

    input_ids = torch.tensor([full_ids], device=device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Log-softmax over vocab
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Sum log-probs of choice tokens (after context)
    # Context tokens end at len(context_ids) (with special tokens offset)
    context_len = len(tokenizer.encode(context, add_special_tokens=True))
    choice_log_prob = 0.0
    num_choice_tokens = 0

    for i in range(context_len - 1, len(full_ids) - 1):
        next_token_id = full_ids[i + 1]
        choice_log_prob += log_probs[0, i, next_token_id].item()
        num_choice_tokens += 1

    # Normalize by number of tokens
    if num_choice_tokens > 0:
        choice_log_prob /= num_choice_tokens

    return choice_log_prob


def evaluate_dataset(
    model,
    tokenizer,
    dataset_name: str,
    num_examples: int = 200,
    max_runtime_minutes: int = 60,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate model on a dataset using 0-shot log-likelihood scoring.
    
    Returns accuracy and detailed results.
    """
    logger.info(f"Evaluating on {dataset_name} ({num_examples} examples)...")

    examples = load_eval_dataset(dataset_name, num_examples, seed)
    start_time = time.time()
    max_runtime_s = max_runtime_minutes * 60

    correct = 0
    total = 0
    per_example = []

    for i, ex in enumerate(examples):
        # Check runtime limit
        elapsed = time.time() - start_time
        if elapsed > max_runtime_s:
            logger.warning(
                f"Runtime limit reached ({max_runtime_minutes}min) "
                f"after {total} examples."
            )
            break

        # Score each choice
        scores = []
        for choice in ex["choices"]:
            try:
                score = score_choice_loglikelihood(
                    model, tokenizer, ex["question"], choice
                )
                scores.append(score)
            except Exception as e:
                logger.warning(f"Scoring error on example {i}: {e}")
                scores.append(float("-inf"))

        # Predict
        predicted = int(np.argmax(scores))
        is_correct = predicted == ex["answer_idx"]

        if is_correct:
            correct += 1
        total += 1

        per_example.append({
            "index": i,
            "correct": is_correct,
            "predicted": predicted,
            "answer": ex["answer_idx"],
            "scores": [round(s, 4) for s in scores],
        })

        if (i + 1) % 50 == 0:
            logger.info(
                f"  Progress: {i+1}/{len(examples)}, "
                f"accuracy so far: {correct}/{total} ({correct/total:.1%})"
            )

    elapsed_total = time.time() - start_time
    accuracy = correct / total if total > 0 else 0.0

    result = {
        "dataset": dataset_name,
        "num_examples_requested": num_examples,
        "num_examples_evaluated": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "runtime_seconds": round(elapsed_total, 1),
        "examples_per_second": round(total / elapsed_total, 2) if elapsed_total > 0 else 0,
        "per_example_results": per_example,
    }

    logger.info(
        f"  {dataset_name}: accuracy={accuracy:.1%} ({correct}/{total}), "
        f"runtime={elapsed_total:.1f}s"
    )

    return result


def run_quality_evaluation(
    model,
    tokenizer,
    dataset_names: List[str] = None,
    num_examples: int = 200,
    max_runtime_minutes: int = 60,
    config_id: str = "baseline",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run full quality evaluation across all configured datasets.
    """
    if dataset_names is None:
        dataset_names = list(QUALITY_CFG.datasets)

    results = {
        "config_id": config_id,
        "num_examples_per_dataset": num_examples,
        "datasets": {},
    }

    for ds_name in dataset_names:
        try:
            ds_result = evaluate_dataset(
                model, tokenizer, ds_name,
                num_examples=num_examples,
                max_runtime_minutes=max_runtime_minutes,
                seed=seed,
            )
            results["datasets"][ds_name] = ds_result
        except Exception as e:
            logger.error(f"Evaluation failed for {ds_name}: {e}")
            results["datasets"][ds_name] = {
                "dataset": ds_name,
                "error": str(e),
            }

    # Aggregate accuracy
    accuracies = []
    for ds_name, ds_result in results["datasets"].items():
        if "accuracy" in ds_result:
            accuracies.append(ds_result["accuracy"])

    if accuracies:
        results["mean_accuracy"] = round(float(np.mean(accuracies)), 4)
    else:
        results["mean_accuracy"] = None

    return results


def compute_quality_retention(
    baseline_results: Dict,
    test_results: Dict,
) -> Dict[str, Any]:
    """
    Compute quality retention compared to baseline.
    
    Returns per-dataset and overall retention metrics.
    """
    retention = {
        "baseline_config": baseline_results.get("config_id", "baseline"),
        "test_config": test_results.get("config_id", "test"),
        "per_dataset": {},
    }

    for ds_name in baseline_results.get("datasets", {}):
        bl = baseline_results["datasets"].get(ds_name, {})
        ts = test_results.get("datasets", {}).get(ds_name, {})

        if "accuracy" in bl and "accuracy" in ts:
            bl_acc = bl["accuracy"]
            ts_acc = ts["accuracy"]
            drop = bl_acc - ts_acc

            retention["per_dataset"][ds_name] = {
                "baseline_accuracy": bl_acc,
                "test_accuracy": ts_acc,
                "accuracy_drop": round(drop, 4),
                "retention_pct": round(
                    (ts_acc / bl_acc * 100) if bl_acc > 0 else 0, 2
                ),
            }

    # Overall
    bl_mean = baseline_results.get("mean_accuracy", 0)
    ts_mean = test_results.get("mean_accuracy", 0)

    if bl_mean and ts_mean:
        retention["overall"] = {
            "baseline_mean_accuracy": bl_mean,
            "test_mean_accuracy": ts_mean,
            "accuracy_drop": round(bl_mean - ts_mean, 4),
            "retention_pct": round(ts_mean / bl_mean * 100 if bl_mean > 0 else 0, 2),
        }

    return retention
