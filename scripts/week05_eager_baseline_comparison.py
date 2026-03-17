"""
Week 05 supplement - Compare all attention backends against an EAGER baseline.

Uses model.generate() for fast token generation, then compares token-level
agreement across backends.
"""

import os
import sys
import json
import gc
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_MODEL_ID, RESULTS_DIR, PROMPTS_DIR
from src.utils import setup_logging, save_results_json, set_seed
from src.model_loader import load_model_4bit
from src.attention_backends import sdpa_backend, check_attention_backends

import torch

logger = setup_logging("llm_inference.eager_baseline")

def log(msg):
    logger.info(msg)
    sys.stdout.flush()


WEEK_DIR = os.path.join(RESULTS_DIR, "week05_attention_opt")
os.makedirs(WEEK_DIR, exist_ok=True)

NUM_NEW_TOKENS = 20
SEED = 42


def _unload(model, tokenizer):
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2)


def _load_prompts():
    with open(os.path.join(PROMPTS_DIR, "fixed_prompts.json")) as f:
        data = json.load(f)
    return [p["text"] for p in data["single_turn"][:3]]


def _generate_tokens(model, tokenizer, prompt, num_tokens):
    """Fast token generation using model.generate()."""
    set_seed(SEED)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=num_tokens,
            do_sample=False,
            temperature=1.0,
        )
    new_token_ids = outputs[0][inputs["input_ids"].shape[-1]:].tolist()
    return new_token_ids


def _compare_tokens(baseline_ids, test_ids):
    n = min(len(baseline_ids), len(test_ids))
    if n == 0:
        return {"steps": 0, "agreement": 0.0}
    matches = sum(1 for a, b in zip(baseline_ids[:n], test_ids[:n]) if a == b)
    first_div = n
    for i in range(n):
        if baseline_ids[i] != test_ids[i]:
            first_div = i
            break
    return {
        "steps": n,
        "matches": matches,
        "agreement": round(matches / n, 4),
        "first_divergence": first_div,
    }


def _run_backend(name, model, tokenizer, prompts, baseline_tokens, ctx_factory=None):
    log(f"  {name}: generating tokens...")
    results_per_prompt = []
    for i, prompt in enumerate(prompts):
        if ctx_factory is not None:
            with ctx_factory():
                gen_ids = _generate_tokens(model, tokenizer, prompt, NUM_NEW_TOKENS)
        else:
            gen_ids = _generate_tokens(model, tokenizer, prompt, NUM_NEW_TOKENS)

        comp = _compare_tokens(baseline_tokens[i], gen_ids)
        comp["prompt_idx"] = i
        results_per_prompt.append(comp)
        log(f"    Prompt {i+1}/{len(prompts)}: {comp['agreement']:.0%} agreement "
            f"({comp['matches']}/{comp['steps']} tokens)")

    rates = [r["agreement"] for r in results_per_prompt]
    agg = {
        "mean_agreement": round(sum(rates) / len(rates), 4),
        "min_agreement": round(min(rates), 4),
        "max_agreement": round(max(rates), 4),
    }
    log(f"  {name} -> mean={agg['mean_agreement']:.1%}, min={agg['min_agreement']:.1%}")
    return {"backend": name, "per_prompt": results_per_prompt, "aggregate": agg}


def main():
    prompts = _load_prompts()
    log(f"Comparing backends against EAGER baseline")
    log(f"  Prompts: {len(prompts)}, New tokens: {NUM_NEW_TOKENS}, Seed: {SEED}")

    # ── Collect eager baseline tokens ────────────────────────────────
    log("")
    log("=" * 60)
    log("Loading EAGER model (no attention optimization)...")
    model, tokenizer = load_model_4bit(DEFAULT_MODEL_ID, attn_implementation="eager")

    baseline_tokens = []
    for i, prompt in enumerate(prompts):
        ids = _generate_tokens(model, tokenizer, prompt, NUM_NEW_TOKENS)
        baseline_tokens.append(ids)
        text = tokenizer.decode(ids, skip_special_tokens=True)
        log(f"  Baseline prompt {i+1}: {len(ids)} tokens -> '{text[:60]}...'")

    # Eager vs Eager (sanity)
    log("")
    eager_result = _run_backend("Eager", model, tokenizer, prompts, baseline_tokens)
    _unload(model, tokenizer)

    # ── SDPA ─────────────────────────────────────────────────────────
    log("")
    log("=" * 60)
    log("Loading SDPA model...")
    model, tokenizer = load_model_4bit(DEFAULT_MODEL_ID, attn_implementation="sdpa")

    sdpa_result = _run_backend("SDPA Default", model, tokenizer, prompts, baseline_tokens)
    sdpa_math_result = _run_backend(
        "SDPA Math", model, tokenizer, prompts, baseline_tokens,
        ctx_factory=lambda: sdpa_backend("math"),
    )
    _unload(model, tokenizer)

    # ── FlashAttention-2 ─────────────────────────────────────────────
    avail = check_attention_backends()
    fa2_result = None
    if avail.get("flash_attn_2_available"):
        log("")
        log("=" * 60)
        log("Loading FlashAttention-2 model...")
        model, tokenizer = load_model_4bit(
            DEFAULT_MODEL_ID, attn_implementation="flash_attention_2"
        )
        fa2_result = _run_backend(
            "FlashAttention-2", model, tokenizer, prompts, baseline_tokens,
        )
        _unload(model, tokenizer)
    else:
        log("FlashAttention-2 not available, skipping.")

    # ── Save & print ─────────────────────────────────────────────────
    all_results = {
        "description": "Token agreement vs EAGER baseline (no attention optimization)",
        "num_prompts": len(prompts),
        "num_new_tokens": NUM_NEW_TOKENS,
        "model": DEFAULT_MODEL_ID,
        "seed": SEED,
        "backends": {},
    }
    for r in [eager_result, sdpa_result, sdpa_math_result, fa2_result]:
        if r:
            all_results["backends"][r["backend"]] = r

    out_path = os.path.join(WEEK_DIR, "correctness_vs_eager_baseline.json")
    save_results_json(all_results, out_path)

    log("")
    log("=" * 72)
    log("RESULTS: Token Agreement vs EAGER Baseline")
    log("=" * 72)
    log(f"{'Backend':<22} {'Mean':>8} {'Min':>8} {'Max':>8}")
    log("-" * 50)
    for r in [eager_result, sdpa_result, sdpa_math_result, fa2_result]:
        if not r:
            continue
        a = r["aggregate"]
        log(f"{r['backend']:<22} {a['mean_agreement']:>7.1%} {a['min_agreement']:>7.1%} {a['max_agreement']:>7.1%}")
    log("=" * 72)
    log(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
