"""
Demo script for live inference across all 4 configurations.

Usage:
    python scripts/demo_inference.py                       # Run all configs with default prompt
    python scripts/demo_inference.py --config sdpa_default # Run a single config
    python scripts/demo_inference.py --interactive          # Interactive prompt mode
    python scripts/demo_inference.py --prompt "Explain quantum computing in one paragraph."
"""

import argparse
import gc
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_MODEL_ID

CONFIGS = {
    "baseline": {
        "label": "Eager (baseline)",
        "attn_implementation": "eager",
        "kv_quant": None,
    },
    "sdpa_default": {
        "label": "SDPA default",
        "attn_implementation": "sdpa",
        "kv_quant": None,
    },
    "flash_attention_2": {
        "label": "FlashAttention-2",
        "attn_implementation": "flash_attention_2",
        "kv_quant": None,
    },
    "kv_int4": {
        "label": "KV-cache INT4",
        "attn_implementation": "eager",
        "kv_quant": "int4",
    },
}

DEFAULT_PROMPT = (
    "Explain the difference between SDPA and FlashAttention-2 "
    "for transformer inference in two sentences."
)


def get_vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def load_model(config_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cfg = CONFIGS[config_name]
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model_kwargs = {
        "quantization_config": quant_config,
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }

    attn = cfg["attn_implementation"]
    if attn:
        model_kwargs["attn_implementation"] = attn

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_ID, **model_kwargs)

    return model, tokenizer, cfg


def run_inference(model, tokenizer, cfg, prompt, max_new_tokens=64):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": None,
    }

    if cfg["kv_quant"]:
        from transformers.cache_utils import QuantizedCache
        nbits = 4 if cfg["kv_quant"] == "int4" else 2
        kv_cache = QuantizedCache(
            backend="quanto",
            config=model.config,
            nbits=nbits,
        )
        gen_kwargs["past_key_values"] = kv_cache

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    new_tokens = output_ids[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    elapsed_ms = (t1 - t0) * 1000
    n_tokens = len(new_tokens)
    throughput = n_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
    peak_vram = get_vram_mb()

    return {
        "response": response,
        "latency_ms": round(elapsed_ms, 1),
        "tokens_generated": n_tokens,
        "throughput_tok_s": round(throughput, 2),
        "peak_vram_mb": round(peak_vram, 1),
    }


def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_comparison_table(results):
    print("\n" + "=" * 90)
    print(f"{'Configuration':<22} {'Latency (ms)':>13} {'Tokens':>7} {'Tok/s':>8} {'VRAM (MB)':>10}")
    print("-" * 90)
    for name, r in results.items():
        label = CONFIGS[name]["label"]
        print(f"{label:<22} {r['latency_ms']:>13,.1f} {r['tokens_generated']:>7} "
              f"{r['throughput_tok_s']:>8.2f} {r['peak_vram_mb']:>10,.1f}")
    print("=" * 90)

    if "baseline" in results and len(results) > 1:
        base_lat = results["baseline"]["latency_ms"]
        print("\nLatency vs. baseline:")
        for name, r in results.items():
            if name == "baseline":
                continue
            delta = (r["latency_ms"] - base_lat) / base_lat * 100
            sign = "+" if delta > 0 else ""
            print(f"  {CONFIGS[name]['label']:<22} {sign}{delta:.1f}%")
    print()


def run_all_configs(prompt, configs_to_run):
    results = {}
    for name in configs_to_run:
        cfg = CONFIGS[name]
        print(f"\n{'─' * 60}")
        print(f"Loading: {cfg['label']} ({name})")
        print(f"{'─' * 60}")

        try:
            model, tokenizer, cfg_dict = load_model(name)
            print(f"  Model loaded. Running inference...")
            result = run_inference(model, tokenizer, cfg_dict, prompt)
            results[name] = result

            print(f"  Latency:    {result['latency_ms']:,.1f} ms")
            print(f"  Throughput: {result['throughput_tok_s']:.2f} tok/s")
            print(f"  VRAM:       {result['peak_vram_mb']:,.1f} MB")
            print(f"  Response:   {result['response'][:120]}...")

            unload_model(model)
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {
                "response": f"ERROR: {e}",
                "latency_ms": 0,
                "tokens_generated": 0,
                "throughput_tok_s": 0,
                "peak_vram_mb": 0,
            }

    return results


def interactive_mode(config_name):
    cfg = CONFIGS[config_name]
    print(f"\nLoading {cfg['label']} for interactive mode...")
    model, tokenizer, cfg_dict = load_model(config_name)
    print("Model loaded. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt or prompt.lower() in ("quit", "exit"):
            print("Exiting.")
            break

        result = run_inference(model, tokenizer, cfg_dict, prompt)
        print(f"\nAssistant: {result['response']}")
        print(f"  [{result['latency_ms']:,.1f} ms | {result['throughput_tok_s']:.2f} tok/s | "
              f"{result['peak_vram_mb']:,.1f} MB VRAM]\n")

    unload_model(model)


def main():
    parser = argparse.ArgumentParser(
        description="Demo inference with multiple attention/KV-cache configurations"
    )
    parser.add_argument("--config", choices=list(CONFIGS.keys()),
                        help="Run a single configuration instead of all")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive prompt mode (requires --config)")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                        help="Prompt to use for non-interactive mode")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Maximum new tokens to generate")
    args = parser.parse_args()

    print("=" * 60)
    print("LLM Inference Demo")
    print(f"Model: {DEFAULT_MODEL_ID}")
    print(f"GPU:   {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    if args.interactive:
        config_name = args.config or "sdpa_default"
        interactive_mode(config_name)
        return

    configs_to_run = [args.config] if args.config else list(CONFIGS.keys())

    print(f"\nPrompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"Configs: {', '.join(configs_to_run)}")

    results = run_all_configs(args.prompt, configs_to_run)
    print_comparison_table(results)

    print("Responses:")
    for name, r in results.items():
        print(f"\n  [{CONFIGS[name]['label']}]")
        print(f"  {r['response']}")


if __name__ == "__main__":
    main()
