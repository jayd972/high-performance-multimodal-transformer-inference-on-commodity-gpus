"""
Convert the in-depth technical project explanation to a styled PDF.
"""

import os, sys
import markdown
from xhtml2pdf import pisa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

OUTPUT_PDF = os.path.join(RESULTS_DIR, "week14_final", "project_explanation.pdf")

MD_TEXT = r"""
# Efficient Transformer Inference on Commodity GPUs
## A Memory-Constrained Systems Study — Technical Deep Dive

**Authors:** Jay Darji (018180851) &amp; Karma Patel (018223517)
**University:** San Jose State University, Spring 2026

---

## 1. Problem Statement

Large Language Models (LLMs) such as GPT-4, LLaMA, and Qwen are autoregressive transformer decoders that generate text token-by-token. A 3-billion-parameter model in FP16 precision requires approximately 6 GB of GPU memory just for weight storage, exceeding the capacity of entry-level discrete GPUs (4 GB VRAM). This project investigates whether **attention kernel optimization** and **KV-cache quantization** can make inference practical on such hardware, specifically an NVIDIA RTX 3050 Ti Laptop GPU (4 GB GDDR6, Ampere SM 8.6).

### Success Criteria

- **Criterion A (Latency):** ≥20% reduction in p95 end-to-end latency at 1024-token prompts
- **Criterion B (Context):** ≥1.5x increase in maximum context length before CUDA OOM
- **Quality Guard-Rail:** ≤2% absolute accuracy drop on a 3-benchmark evaluation suite

---

## 2. Model Architecture and Quantization

### 2.1 Target Model: Qwen2.5-3B-Instruct

| Parameter | Value |
|-----------|-------|
| Total parameters | 3.09 billion |
| Layers (L) | 36 |
| Hidden dimension (H) | 2048 |
| Query heads (N_Q) | 16 |
| Key-Value heads (N_KV) | 2 (Grouped Query Attention) |
| Head dimension (d_head) | 128 |
| Vocabulary size | 151,936 |
| FP16 weight footprint | ~6.0 GB |
| NF4 weight footprint | **1,917 MB (1.87 GB)** |

### 2.2 4-bit NF4 Quantization

Standard FP16 stores each parameter in 2 bytes. **NormalFloat 4-bit (NF4)** quantization, implemented by the bitsandbytes library, compresses each parameter to 0.5 bytes using a non-uniform quantization grid optimized for normally-distributed weights.

The process:

1. **Block quantization:** Weights are divided into blocks of 64 elements
2. **Absmax scaling:** Each block is scaled by its maximum absolute value
3. **NF4 mapping:** Each scaled value is mapped to the nearest of 16 NF4 levels (4 bits)
4. **Double quantization:** The scaling factors themselves are quantized to FP8, saving additional memory

This reduces the weight footprint from ~6 GB to 1.87 GB — a **3.2x compression** — while preserving model quality. The tradeoff: at inference time, each weight must be **dequantized** back to FP16 before matrix multiplication, adding ~14% runtime overhead (measured via profiling).

### 2.3 Grouped Query Attention (GQA)

Standard multi-head attention (MHA) uses N_Q query heads and N_Q key-value heads (1:1 ratio). GQA reduces the KV heads to N_KV < N_Q, sharing each KV head across N_Q/N_KV query heads.

For Qwen2.5-3B: 16 query heads share 2 KV heads (8:1 ratio). This reduces KV-cache memory by 8x compared to standard MHA, but introduces compatibility constraints with certain attention kernels that expect matching head counts.

---

## 3. Transformer Inference: How It Works

### 3.1 The Two Phases

Autoregressive inference proceeds in two distinct phases:

**Phase 1 — Prefill (Prompt Processing):**
The entire input prompt is processed in a single forward pass. Attention is computed over all prompt tokens simultaneously. Computational complexity: O(N^2 * d) for N prompt tokens, dominated by the attention matrix computation.

**Phase 2 — Decode (Token Generation):**
Tokens are generated one at a time. Each new token attends to all previous tokens (prompt + previously generated). The KV-cache stores the Key and Value projections from all previous tokens so they don't need to be recomputed. Per-step complexity: O(N * d) where N is the current sequence length.

### 3.2 Measured Prefill vs Decode

We separated these phases using linear regression across prompt lengths. Since decode cost is approximately constant (fixed 64 output tokens), varying the prompt length isolates the prefill component:

T_total = r_prefill * N_prompt + r_decode * N_output

| Configuration | Prefill Rate (ms/tok) | Decode Rate (ms/tok) | Decode Speedup |
|---------------|-----------------------|----------------------|----------------|
| baseline (eager) | 0.39 | 168.1 | — |
| sdpa_default | 0.98 | 99.9 | **1.68x** |
| flash_attention_2 | 0.80 | 106.2 | **1.58x** |
| kv_int4 | 0.94 | 113.4 | **1.48x** |

**Key insight:** Decode dominates 91-99% of total inference time. At 1024 prompt tokens with 64 output tokens:

- Baseline: prefill = 403 ms (3.6%), decode = 10,760 ms (96.4%)
- SDPA default: prefill = 1,002 ms (13.5%), decode = 6,395 ms (86.5%)

SDPA's 33% total latency reduction is primarily a **decode-phase speedup** — the optimized kernel reduces per-token decode from 168 ms to 100 ms.

The linear model fits measured data with less than 2.2% error across all configurations and prompt lengths, validating the two-phase decomposition.

---

## 4. Attention Backend Optimization

### 4.1 The Attention Computation

The core attention operation for each head is:

Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_head)) * V

Where Q, K, V are the query, key, and value matrices. For a sequence of length T with head dimension d:

- Q * K^T produces a T x T attention weight matrix
- This matrix is stored in GPU HBM (High Bandwidth Memory)
- Memory: O(T^2) per head per layer

For Qwen2.5-3B at T=1024: 16 heads * 36 layers * 1024^2 * 4 bytes (FP32) = **2.4 GB** just for attention matrices. This is why eager attention hits OOM at ~6,784 tokens.

### 4.2 Backends Tested

**Eager Attention (Baseline):**
Standard PyTorch implementation. Materializes the full T x T attention matrix in HBM. Memory: O(T^2). Simple but memory-hungry.

**SDPA Default:**
PyTorch's `torch.nn.functional.scaled_dot_product_attention` — a unified API that dispatches to the best available backend at runtime. On Windows with CUDA 12.4, this dispatches to the math-mode kernel, which applies fused softmax and avoids some intermediate allocations. Memory: still O(T^2) but with better constant factors.

**FlashAttention-2:**
A tiled attention algorithm that never materializes the full T x T matrix. Instead:

1. Divide Q, K, V into blocks that fit in GPU SRAM (shared memory)
2. Compute attention for each block pair
3. Use online softmax normalization to combine partial results
4. Write only the final output to HBM

Memory: O(T) — linear in sequence length. This is the fundamental reason FlashAttention-2 extends context length.

**SDPA Math:**
Explicitly forces the math-mode SDP backend. In practice, identical to SDPA default on our hardware.

**Excluded backends:**
- Flash SDP: Not available in pre-compiled PyTorch Windows wheels
- Memory-efficient SDP (xFormers): Incompatible with GQA 16:2 head ratio — returns "No available kernel"

### 4.3 Results

| Configuration | p95 Latency (ms) | Throughput (tok/s) | Peak VRAM (MB) | vs Baseline |
|---------------|-------------------|--------------------|--------------------|-------------|
| baseline (eager) | 11,209 | 5.73 | 2,152 | — |
| sdpa_default | 7,509 | 8.64 | 2,203 | **-33.0%** |
| flash_attention_2 | 7,641 | 8.40 | 2,115 | -31.8% |
| kv_int4 | 8,091 | 7.80 | 2,136 | -27.8% |

All measurements at 1024-token prompts, batch size 1, 64 output tokens, 5 timed runs after 2 warmup runs, greedy decoding (temperature=0, seed=42).

### 4.4 OOM Thresholds

| Configuration | Max Context (tokens) | Improvement |
|---------------|----------------------|-------------|
| baseline (eager) | 6,784 | — |
| sdpa_default | 7,040 | +3.8% |
| flash_attention_2 | 8,064 | **+18.9%** |
| kv_int4 | 6,784 | 0% |

FlashAttention-2's O(T) memory scaling is clearly visible: at 8,064 tokens, peak VRAM is only 2,898 MB. The baseline would require ~8 GB at the same length (extrapolated from its quadratic growth curve).

---

## 5. KV-Cache Quantization

### 5.1 The KV-Cache Memory Problem

During autoregressive generation, each layer stores Key and Value tensors for all previous tokens. For Qwen2.5-3B in FP16:

KV_bytes_per_token = 2 (K+V) * L * N_KV * d_head * 2 (bytes)
                   = 2 * 36 * 2 * 128 * 2
                   = 36,864 bytes = 0.0352 MB per token

At 8,000 tokens: 0.0352 * 8000 = 281 MB for KV-cache alone.

### 5.2 Quantization via optimum-quanto

The optimum-quanto library quantizes KV-cache tensors at generation time:

- **INT4:** Each FP16 value mapped to 4-bit integer with per-group scaling factors. Effective size: ~0.625 bytes/element (0.5 bytes + scale overhead)
- **INT2:** Each FP16 value mapped to 2-bit integer with per-group scaling factors. Effective size: ~0.375 bytes/element (0.25 bytes + scale overhead)

Note: INT8 is NOT supported by the quanto backend despite being listed in some documentation.

### 5.3 Measured Per-Token Memory Growth

| KV Precision | Analytical (MB/tok) | Measured (MB/tok) | Ratio |
|--------------|---------------------|-------------------|-------|
| FP16 | 0.0352 | 0.181 | 5.1x |
| INT4 | 0.0110 | 0.157 | 14.3x |
| INT2 | 0.0066 | 0.153 | 23.2x |

The measured per-token growth is 5-23x larger than the pure KV-cache prediction. This gap comes from:

- **Activation buffers** that scale with sequence length (intermediate tensors in each layer)
- **Attention intermediate tensors** (Q*K^T products, softmax outputs)
- **PyTorch memory allocator fragmentation** (CUDA caching allocator over-allocates)
- **Gradient-like bookkeeping** tensors maintained by the framework

This analysis reveals that the KV-cache is a **minor contributor** to per-token VRAM growth at these sequence lengths — attention buffers and allocator overhead dominate.

### 5.4 Tradeoff Analysis

| Prompt Length | KV Type | VRAM Saved | Latency Change |
|---------------|---------|------------|----------------|
| 128 | INT4 | 3.1 MB (0.2%) | +19.5% |
| 128 | INT2 | 3.7 MB (0.2%) | +25.6% |
| 512 | INT4 | 12.5 MB (0.6%) | **-1.6%** |
| 512 | INT2 | 14.7 MB (0.7%) | +2.1% |
| 1024 | INT4 | 23.9 MB (1.1%) | +2.7% |
| 1024 | INT2 | 28.1 MB (1.3%) | +4.3% |

At short contexts, the overhead of quantization/dequantization outweighs the memory benefit. At longer contexts, the savings compound: at 8,000 tokens, INT4 would save ~192 MB.

### 5.5 Correctness Verification

| KV Precision | Top-1 Agreement | Max Logit Difference |
|--------------|-----------------|----------------------|
| INT4 | **100.0%** | 0.0 |
| INT2 | **100.0%** | 0.0 |

Under greedy decoding, both quantization levels produce **bitwise-identical token selections** to the FP16 baseline. The quantization error is below the threshold that would alter any argmax decision.

### 5.6 Final Selection

**INT4 selected** as the recommended KV-cache configuration:

- Average VRAM savings: 0.6% (grows with context length)
- Average latency overhead: +6.9%
- Zero quality degradation
- INT2 rejected: +10.7% latency overhead for only marginal additional savings

---

## 6. Analytical Memory Model

### 6.1 VRAM Decomposition

We derive a first-principles model of GPU memory consumption:

**VRAM_total = W_weights + KV_cache(T) + Attn_buffer(T) + Overhead**

Where:

- **W_weights** = 1,917 MB (fixed, NF4 quantized)
- **KV_cache(T)** = 2 * L * N_KV * d_head * bytes_per_element * T (linear in T)
- **Attn_buffer_eager(T)** = L * N_Q * T^2 * 4 bytes (quadratic in T for eager)
- **Attn_buffer_flash(T)** = L * N_Q * T * d_head * 2 bytes (linear in T for FlashAttention)
- **Overhead** = ~100 MB (CUDA context, PyTorch runtime, allocator)

### 6.2 Predicted vs Measured

| Seq Length | Predicted Eager (MB) | Predicted Flash (MB) | Measured Eager (MB) | Measured Flash (MB) |
|------------|----------------------|----------------------|---------------------|---------------------|
| 128 | 2,105 | 2,051 | 2,027 | 2,034 |
| 512 | 2,449 | 2,065 | 2,070 | 2,069 |
| 1024 | 3,580 | 2,089 | 2,152 | 2,115 |
| 4096 | 39,013 | 2,213 | — (OOM) | — |

The model correctly predicts the **qualitative behavior**: eager grows quadratically (explodes at long contexts), FlashAttention grows linearly (stays manageable). The quantitative gap at short contexts arises because PyTorch's CUDA caching allocator reuses memory pools, making measured VRAM lower than the theoretical maximum.

### 6.3 Why This Matters

The analytical model explains **why** FlashAttention-2 extends context by 19%: at 8,064 tokens, the eager attention buffer alone would require ~15 GB (far exceeding 4 GB VRAM), while FlashAttention's tiled approach keeps total VRAM under 3 GB. The bottleneck shifts from attention buffers to KV-cache and weight storage.

---

## 7. System Bottleneck Analysis

### 7.1 Runtime Profiling

Using PyTorch's built-in profiler with CUDA event timing, we decomposed kernel execution time:

| Component | CUDA Time (ms) | % of Kernel Time |
|-----------|-----------------|------------------|
| Attention (SDPA math) | 11,849 | **32.0%** |
| Memory Ops (copy, transpose, dtype conversion) | 10,401 | **28.1%** |
| 4-bit Dequantization (bitsandbytes) | 5,159 | **13.9%** |
| Other (embedding, softmax, layernorm, etc.) | 7,731 | 20.9% |
| Linear / MatMul | 1,871 | 5.1% |

**Total non-wrapper kernel time:** 37,012 ms for 64 generated tokens.

### 7.2 Key Findings

**Attention is the single largest component (32%)** — validating our focus on attention optimization. The SDPA math kernel (`aten::_scaled_dot_product_attention_math`) is called 2,304 times (36 layers * 64 decode steps), consuming 5.9 seconds of GPU time.

**Memory operations are surprisingly expensive (28%)** — `aten::to` (dtype conversion), `aten::transpose`, and `aten::t` (tensor reshaping) collectively consume 10.4 seconds. This overhead comes from:

- FP16-to-FP32 conversions for attention computation
- Tensor layout transformations between contiguous and strided formats
- KV-cache tensor concatenation at each decode step

**4-bit dequantization adds 14% overhead** — `bitsandbytes::dequantize_blockwise` (2.2s) and `bitsandbytes::gemv_4bit` (3.0s) are the cost of fitting the model in 4 GB. Each of the 252 linear layers (7 per transformer block * 36 blocks) must dequantize weights from NF4 to FP16 before every forward pass.

**Linear/MatMul is only 5%** — because the fused `gemv_4bit` kernel combines dequantization with matrix-vector multiplication, the separate `aten::matmul` calls are relatively few.

### 7.3 VRAM Breakdown at 1024 Tokens (Analytical)

| Component | MB | % of Total |
|-----------|-----|-----------|
| Model Weights (NF4) | 1,917 | 41.2% |
| Attention Buffers (eager, O(T^2)) | 2,601 | 55.9% |
| KV Cache (FP16) | 38 | 0.8% |
| Framework Overhead | 100 | 2.1% |
| **Total Predicted** | **4,656** | 100% |

At 1024 tokens, the analytical model predicts 4,656 MB — exceeding the 4 GB physical limit. In practice, PyTorch's caching allocator and kernel fusion reduce actual peak usage to ~2,152 MB because attention buffers are allocated and freed within each layer's forward pass (they don't all exist simultaneously).

---

## 8. Quality Evaluation

### 8.1 Methodology

Quality was assessed using 0-shot log-likelihood evaluation: for each benchmark example, the model scores every candidate answer by computing the sum of log-probabilities of the answer tokens conditioned on the question. The highest-scoring candidate is selected. No gradient computation or fine-tuning is involved.

### 8.2 Results

| Benchmark | Task Type | Examples | Correct | Accuracy | Runtime |
|-----------|-----------|----------|---------|----------|---------|
| ARC-Easy | Science QA (4-choice) | 200 | 125 | 62.5% | 10.5 min |
| BoolQ | Boolean QA (yes/no) | 113/200 | 68 | 60.2% | 121 min |
| HellaSwag | Commonsense (4-choice) | 200 | 138 | 69.0% | 16.2 min |
| **Mean** | — | — | — | **63.9%** | — |

BoolQ completed only 113/200 examples due to a 60-minute runtime limit per configuration. At ~32 seconds per example (each example scores multiple candidate completions), the full 200 would require ~107 minutes.

The 63.9% mean accuracy is consistent with published benchmarks for 3B-parameter models at 4-bit quantization, confirming that NF4 compression preserves model capability.

---

## 9. Combined Configuration Failure Analysis

### 9.1 What Was Attempted

Combining attention optimization (SDPA/FlashAttention-2) with KV-cache quantization (INT4/INT2) to achieve both latency reduction and memory savings simultaneously.

### 9.2 What Failed

| Attention Backend | KV Quantization | Status | Error |
|-------------------|-----------------|--------|-------|
| mem_efficient SDP | INT4 | **FAILED** | No available kernel. Aborting execution. |
| mem_efficient SDP | INT2 | **FAILED** | No available kernel. Aborting execution. |

### 9.3 Root Cause

The failure is a **kernel dispatch issue**, not an algorithmic limitation:

1. Qwen2.5-3B's GQA uses 16 query heads and 2 KV heads (8:1 ratio)
2. The memory-efficient attention kernel (from xFormers) requires specific head-count ratios to select a valid CUDA kernel
3. When KV-cache tensors are quantized, their dtype changes from FP16 to a custom quantized format
4. The combination of non-standard head ratio + non-standard dtype exceeds the dispatch logic's supported configurations

This is expected to be resolved in future PyTorch/FlashAttention releases as GQA becomes more prevalent.

---

## 10. Experimental Infrastructure

### 10.1 Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU |
| VRAM | 4 GB GDDR6, 192-bit bus |
| Compute Capability | 8.6 (Ampere architecture) |
| CUDA Cores | 2560 |
| Memory Bandwidth | 192 GB/s |
| CPU | Intel Core (16 logical cores) |
| System RAM | 15.2 GB DDR4 |
| OS | Windows 10 (build 26200) |

### 10.2 Software Stack

| Package | Version |
|---------|---------|
| Python | 3.12.9 |
| PyTorch | 2.6.0+cu124 |
| CUDA Toolkit | 12.4 |
| Transformers | 4.57.6 |
| bitsandbytes | 0.49.1 |
| accelerate | 1.12.0 |
| optimum-quanto | 0.2.7 |
| flash-attn | 2.7.4+cu124 (pre-built Windows wheel) |
| datasets | 4.5.0 |

### 10.3 Benchmark Protocol

| Parameter | Value |
|-----------|-------|
| Prompt lengths | 128, 512, 1024 tokens |
| Max new tokens | 64 |
| Batch size | 1 |
| Decoding | Greedy (do_sample=False, temperature=0) |
| Random seed | 42 |
| Warmup runs | 2 (discarded) |
| Timed runs | 5 |
| VRAM monitoring | pynvml at 100ms intervals |

### 10.4 Confounding Factors

**Thermal throttling:** The laptop GPU throttles under sustained load, causing non-monotonic latency curves at longer contexts. Mitigated with warmup runs but not fully eliminable.

**Windows platform limitations:** Flash SDP backend unavailable in pre-compiled PyTorch wheels. Memory-efficient SDP incompatible with GQA. CUDA memory management differs from Linux.

---

## 11. Results Summary

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| A: Latency | ≥20% p95 reduction | 33.0% (SDPA default) | **MET** |
| B: Context | ≥1.5x max context | 1.19x (FlashAttention-2) | Not met |
| Quality | ≤2% accuracy drop | 0% drop (63.9% mean) | **MET** |

### Practical Recommendations

| Workload | Configuration | Rationale |
|----------|---------------|-----------|
| Latency-sensitive | SDPA default | 33% p95 reduction, 1.68x decode speedup |
| Long-context | FlashAttention-2 | 19% more context, O(T) memory scaling |
| VRAM-constrained | Add KV-cache INT4 | 13% per-token savings, zero quality loss |
| General purpose | SDPA default | Best speed/compatibility balance |

### Future Work

- **Desktop GPU validation:** RTX 3060/4060 with better thermals
- **Larger models:** Qwen2.5-7B under the same 4 GB constraint
- **Linux deployment:** Unlock flash SDP + memory-efficient backends
- **Batched inference:** Throughput for serving scenarios (batch > 1)
- **Speculative decoding:** Further latency reduction via draft models

---

## 12. Project Deliverables

| Category | Count | Description |
|----------|-------|-------------|
| Weekly experiment scripts | 10 | Weeks 1-10 automated experiments |
| Utility scripts | 9 | Plotting, presentation, demo, analysis |
| Source modules | 9 | Model loading, benchmarking, profiling, etc. |
| Result directories | 14 | Weeks 1-14 outputs |
| Publication-quality plots | 20 | Baseline, attention, KV-cache, final, advanced |
| Data tables (JSON) | 50+ | Raw measurements, analyses, summaries |
| Final report | 571 lines | Complete academic paper with 12 figures, 10 tables |
| Presentation | 9 slides | Professional PowerPoint |
| Demo script | 1 | Interactive inference with 4 configurations |
| Reproducibility package | Complete | Locked deps, hardware specs, benchmark protocol |
"""

CSS = """
@page {
    size: letter;
    margin: 0.8in 0.75in;
}

body {
    font-family: "Helvetica", "Arial", sans-serif;
    font-size: 10pt;
    line-height: 1.45;
    color: #1a1a1a;
}

h1 {
    font-size: 18pt;
    text-align: center;
    margin-top: 0;
    margin-bottom: 2pt;
    color: #1a3a5c;
}

h2 {
    font-size: 13pt;
    margin-top: 14pt;
    margin-bottom: 5pt;
    color: #1a3a5c;
    border-bottom: 2px solid #1a3a5c;
    padding-bottom: 2pt;
}

h3 {
    font-size: 11pt;
    margin-top: 10pt;
    margin-bottom: 3pt;
    color: #2c5282;
}

p {
    margin-top: 2pt;
    margin-bottom: 5pt;
    text-align: justify;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 5pt;
    margin-bottom: 7pt;
    font-size: 9pt;
}

th {
    background-color: #1a3a5c;
    color: white;
    padding: 4pt 5pt;
    text-align: center;
    font-weight: bold;
    border: 1px solid #1a3a5c;
}

td {
    padding: 3pt 5pt;
    border: 1px solid #ddd;
    text-align: center;
}

tr:nth-child(even) td {
    background-color: #f0f4f8;
}

strong {
    color: #1a1a1a;
}

code {
    font-family: "Courier New", monospace;
    font-size: 8.5pt;
    background-color: #f0f4f8;
    padding: 1pt 2pt;
}

hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 10pt 0;
}

ul, ol {
    margin-top: 2pt;
    margin-bottom: 4pt;
    padding-left: 16pt;
}

li {
    margin-bottom: 2pt;
}
"""


def convert():
    html_body = markdown.markdown(MD_TEXT, extensions=["tables", "fenced_code", "sane_lists"])

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{CSS}</style></head>
<body>
{html_body}
</body>
</html>"""

    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)
    with open(OUTPUT_PDF, "wb") as pdf_file:
        status = pisa.CreatePDF(html, dest=pdf_file)

    if status.err:
        print(f"ERROR: {status.err} errors")
        return False

    size_kb = os.path.getsize(OUTPUT_PDF) / 1024
    print(f"PDF saved: {OUTPUT_PDF}")
    print(f"Size: {size_kb:.0f} KB")
    return True


if __name__ == "__main__":
    success = convert()
    sys.exit(0 if success else 1)
