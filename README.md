# MSA: Memory Sparse Attention
*A scalable, end-to-end trainable latent-memory framework for 100M-token contexts*

[**arXiv**](https://arxiv.org/abs/2603.23516) • [**Zenodo**](https://zenodo.org/records/19103670)

[![Models](https://img.shields.io/badge/HuggingFace-MSA--4B-yellow?logo=huggingface)](https://huggingface.co/EverMind-AI/MSA-4B) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![EverMind](https://img.shields.io/badge/EverMind-Organization-blue?logo=github)](https://github.com/EverMind-AI) [![EverOS](https://img.shields.io/badge/EverOS-Repo-green?logo=github)](https://github.com/EverMind-AI/everos)

[中文](./README_ZH.md)

## 📝 Abstract

Long-term memory is essential for general intelligence, yet the **full attention** bottleneck constrains most LLMs' **effective context length** to **128K–1M**. Existing attempts，hybrid linear attention, fixed-size state memory (e.g., RNNs), and external storage like **RAG/agents**，either suffer rapid precision decay and latency growth at extreme scales, lack end-to-end differentiability or dynamic memory maintenance, or require complex pipelines. We present **Memory Sparse Attention (MSA)**: an **end-to-end trainable, scalable sparse** **latent-state memory** framework. Core ideas include:

- **Scalable sparse attention** + **document-wise RoPE** (parallel/global) achieving **near-linear complexity** in both training and inference;
- **KV cache compression** with a **Memory Parallel** inference engine to deliver **100M token** throughput on **2×A800** GPUs;
- **Memory Interleave** for multi-round, multi-hop reasoning across scattered memory segments.

On long-context QA and NIAH (Needle-in-a-Haystack) benchmarks, **MSA** surpasses same-backbone RAG, best-of-breed RAG stacks, and leading long-context models. Across an unprecedented **16K→100M token** range, MSA shows **< 9%** degradation, suggesting a practical path to **decouple memory capacity from reasoning**.

> **Scaling from 16K→100M tokens**: MSA fuses top-k selection with sparse attention to remain end-to-end differentiable while allowing document decoupling at inference. On MS MARCO, MSA sustains **<9%** degradation and exhibits strong extrapolation.
> *Some baseline curves end early due to their context limits.*

![Figure 1: Scaling curve 16K→100M tokens](./assets/fig1_scaling.png)
*Figure 1: MSA scalability under extreme-long contexts*

---

## ✨ Key Contributions

- **Memory-Sparse Attention (MSA)**: an **end-to-end trainable**, **scalable sparse attention** layer with **document-wise RoPE**, realizing **O(L)** complexity and **<9%** degradation from **16K→100M tokens**.
- **KV cache compression + Memory Parallel**: tiered storage (GPU-resident routing keys, CPU content K/V), distributed scoring, and on-demand transfers to enable **100M-token** inference on **2×A800**.
- **Memory Interleave**: adaptive alternating "generative retrieval → context expansion → generation," significantly boosting **multi-hop** reasoning across documents.
- **Comprehensive evaluation**: MSA outperforms same-backbone RAG, best-of-breed RAG pipelines, and top long-context models on long-context QA and NIAH, showing superior **stability** and **accuracy** at scale.

---

## 🧩 Overall Design

### Architecture

MSA integrates **retrieval and generation** into a single differentiable loop. Document latent states (**K/V/Kᵣ**) are **chunk-mean pooled** for compression. A **router projector** computes relevance via cosine similarity (mean-pooled over heads, then token-wise max), selects **Top‑k documents**, then concatenates their **compressed K/V** with the query's local **K/V** for autoregressive decoding. Routing applies only to **upper layers**; lower layers keep **independent document processing** for hierarchical alignment.

- **Parallel (document-wise) RoPE**: Each document resets positions from 0, preventing position drift between **train-short** and **infer-long**, enabling 64k training to extrapolate to 100M.
- **Global RoPE (active context)**: The query's starting index is **offset by k** (Top‑k retrieved blocks), preserving causal ordering: *background → query → generation*.

**Figure 2: MSA layer (sparse attention + document-wise RoPE)**

![Figure 2: MSA layer](./assets/fig2_msa_layer.png)
*Figure 2: Memory-Sparse Attention layer and parallel/global RoPE*

---

### Inference Pipeline

MSA uses a **three-stage** pipeline (Fig. 3):

1. **Global Memory Encoding (offline)**: forward over the corpus to cache chunk-pooled **(K̄, V̄, K̄ᵣ)**.
2. **Online Routing & Context Assembly**: project query to **Qᵣ**, match with **K̄ᵣ** to pick **Top‑k**, then load only the selected **K̄/V̄** and concatenate with local context.
3. **Sparse Generation**: autoregress over the **sparse context**.

**Memory Parallel** shards **K̄ᵣ** across GPUs (query broadcast → local scoring → global reduce). Content **K̄/V̄** stays in host DRAM and is **asynchronously fetched** when selected—balancing **VRAM** and **throughput** for **100M-token** deployment.

**Figure 3: Three-stage inference & Memory Interleave**

![Figure 3: Inference](./assets/fig3_inference.png)
*Figure 3: Offline encoding → online routing → sparse generation; optional multi-round interleave for multi-hop*

---

## 🚀 Results

> **Setup**
> **QA**: 9 datasets (MS MARCO v1, NQ, DuReader, TriviaQA(10M), NarrativeQA, PopQA, 2WikiMultiHopQA, HotpotQA, MuSiQue), memory banks **277K→10M tokens**, metric: **LLM judge (0–5)**.
> **NIAH (RULER)**: 8 subtasks, **32K→1M tokens**, report average accuracy.
> **Backbone**: Qwen3‑4B‑Instruct‑2507. Compare to same-backbone RAG and best-of-breed RAG stacks (KaLMv2 + large generators, optional reranker).

### Table 1: MSA vs same-backbone RAG (Qwen3‑4B)

**Summary**: Average **3.760**, improving over standard RAG (**+16.0%**), RAG+rerank (**+11.5%**), and HippoRAG2 (**+14.8%**) using their best@k; MSA leads on all but NarrativeQA within the same-backbone group.


| Dataset | Tokens | Qwen3-4B R@1 | R@5 | R@10 | Qwen3-4B (RR) R@1 | R@5 | R@10 | HippoRAG2 R@1 | R@5 | R@10 | MSA (adaptive) |
|---------|--------|--------------|------|-------|----------------------|------|-------|------------------|------|--------|----------------|
| MS MARCO v1 | 7.34M | 2.893 | 3.011 | 3.005 | 2.934 | <u>3.032</u> | 3.017 | 2.676 | 3.005 | 3.019 | **4.141** |
| Natural Questions | 1.47M | 3.452 | 3.374 | 3.297 | <u>3.494</u> | 3.408 | 3.385 | 3.338 | 3.389 | 3.374 | **3.545** |
| DuReader | 277K | 3.726 | 3.579 | 3.594 | <u>3.848</u> | 3.618 | 3.607 | 2.941 | 3.485 | 3.415 | **4.155** |
| TriviaQA (10M) | 10M | 4.133 | 4.414 | 4.273 | 4.313 | 4.375 | 4.391 | 4.188 | <u>4.430</u> | 4.367 | **4.621** |
| NarrativeQA | 538K | 1.611 | 2.567 | 2.860 | **3.638** | 3.492 | <u>3.536</u> | 1.959 | 2.628 | 2.655 | 3.395 |
| PopQA | 1.18M | 2.959 | 3.273 | 3.299 | <u>3.315</u> | 3.264 | 3.266 | 3.111 | 3.249 | 3.249 | **3.433** |
| 2WikiMultiHopQA | 722K | 1.065 | 3.055 | 3.136 | 1.187 | 3.057 | 3.159 | 1.045 | 3.180 | <u>3.330</u> | **4.280** |
| HotpotQA | 1.35M | 2.252 | 3.582 | 3.787 | 2.642 | 3.990 | <u>4.022</u> | 3.230 | 3.770 | 3.970 | **4.061** |
| MuSiQue | 1.41M | 0.936 | 1.752 | 1.928 | 1.144 | 1.960 | 1.965 | 1.020 | 1.907 | <u>2.095</u> | **2.211** |
| **Average** | — | 2.559 | 3.179 | 3.242 | 2.946 | 3.355 | <u>3.372</u> | 2.612 | 3.227 | 3.275 | **3.760** |

*Table 1: Same-backbone RAG vs MSA (@1/@5/@10 vs MSA @adaptive)*

---

### Table 2: MSA vs best-of-breed RAG (large backbones)

**Summary**: Against **KaLMv2+Qwen3‑235B** and **KaLMv2+Llama‑3.3‑70B** (w/ and w/o reranking), MSA achieves the best score on **4/9** datasets and an average **3.760**, with relative gains of **+7.2%**, **+5.0%**, **+10.7%**, and **+5.4%** over the strongest configurations respectively. Gaps on a few datasets (e.g., MuSiQue) are largely attributable to parameter-count and intrinsic reasoning capacity.


| Dataset | KaLMv2 + Qwen3‑235B R@1 | R@5 | R@10 | Qwen3‑235B (RR) R@1 | R@5 | R@10 | KaLMv2 + Llama‑3.3 R@1 | R@5 | R@10 | Llama‑3.3 (RR) R@1 | R@5 | R@10 | MSA (adaptive) |
|---------|---------------------------|------|-------|--------------------------|------|-------|------------------------------|------|--------|-------------------------|------|--------|----------------|
| MS MARCO v1 | 2.846 | <u>3.028</u> | 3.027 | 2.886 | 3.020 | 2.995 | 2.649 | 2.904 | 2.919 | 2.881 | 2.955 | 2.952 | **4.141** |
| Natural Questions | <u>3.711</u> | 3.670 | 3.694 | 3.621 | 3.610 | 3.645 | 3.675 | 3.674 | 3.662 | **3.756** | 3.665 | 3.647 | 3.545 |
| DuReader | 4.044 | 3.991 | 3.978 | 3.973 | 3.932 | 3.891 | <u>4.051</u> | 3.846 | 3.742 | 3.967 | 3.776 | 3.780 | **4.155** |
| TriviaQA (10M) | 4.367 | 4.656 | 4.578 | 4.492 | 4.320 | 4.555 | 4.273 | **4.740** | <u>4.719</u> | 4.547 | 4.703 | 4.695 | 4.621 |
| NarrativeQA | 1.413 | 2.130 | 2.427 | 3.212 | **3.427** | 3.375 | 1.290 | 2.123 | 2.382 | 3.150 | 3.263 | 3.317 | <u>3.395</u> |
| PopQA | 2.810 | 3.347 | <u>3.396</u> | 3.268 | 3.380 | 3.376 | 2.787 | 3.298 | 3.305 | 3.337 | 3.384 | 3.362 | **3.433** |
| 2WikiMultiHopQA | 2.646 | 3.579 | 3.582 | 1.855 | 3.381 | <u>3.583</u> | 1.339 | 3.263 | 3.445 | 1.651 | 3.332 | 3.541 | **4.280** |
| HotpotQA | 3.497 | 4.090 | **4.225** | 3.341 | 4.141 | 4.194 | 3.070 | 3.896 | 4.127 | 3.428 | 4.145 | <u>4.203</u> | 4.061 |
| MuSiQue | 1.988 | 2.462 | **2.647** | 1.801 | 2.522 | 2.605 | 1.704 | 2.317 | 2.258 | 1.895 | 2.462 | <u>2.614</u> | 2.211 |
| **Average** | 3.036 | 3.439 | 3.506 | 3.161 | 3.526 | <u>3.580</u> | 2.760 | 3.340 | 3.396 | 3.179 | 3.521 | 3.568 | **3.760** |

*Table 2: SOTA RAG stacks (strong retriever + large generator + optional reranker) vs MSA*

---

### Figure 4: RULER NIAH stability (32K→1M)

**Summary**: MSA maintains **94.84%** at **1M tokens**. The unmodified backbone collapses beyond **128K** (down to **24.69% @1M**). Hybrid linear-attention long-context models degrade noticeably at **≥128K/256K**. External-memory agents (e.g., RL‑MemoryAgent‑14B) remain stable but are weaker in **absolute accuracy** and show steeper decay than MSA.

![Figure 4: RULER NIAH 32K→1M](./assets/fig4_ruler_niah.png)
*Figure 4: Accuracy vs context length (higher is better)*

---

## Implementation Notes

- **Training**: 158.95B-token continuous pretraining with **auxiliary routing loss**, followed by two-stage SFT (**8k→64k** curriculum).
- **Ablations** (paper Table 4): curriculum extension, Memory Interleave, continuous pretraining, and injecting original text all contribute substantially; removing them causes **5%–37%** drops depending on task.

---

## 🚀 Quick Start

> For full details (project structure, supported benchmarks, etc.), see [**QUICK_START.md**](./QUICK_START.md).

**1. Install**

```bash
conda create -n msa python=3.12 -y && conda activate msa
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

**2. Download model**

```bash
mkdir ckpt
huggingface-cli download --resume-download EverMind-AI/MSA-4B --local-dir ckpt/MSA-4B
```

**3. Download benchmarks**

Benchmark data is hosted on [EverMind-AI/MSA-RAG-BENCHMARKS](https://huggingface.co/datasets/EverMind-AI/MSA-RAG-BENCHMARKS) and will be automatically downloaded to `data/` on first run.

**4. Run**

```bash
# Run inference on benchmarks
bash scripts/run_benchmarks.sh eval_benchmark

# Compute LLM-based scores
bash scripts/calculate_llm_score.sh eval_benchmark
```

---

## Citation

```bibtex
@misc{chen2026msamemorysparseattention,
      title={MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens},
      author={Yu Chen and Runkai Chen and Sheng Yi and Xinda Zhao and Xiaohong Li and Jianjin Zhang and Jun Sun and Chuanrui Hu and Yunyun Han and Lidong Bing and Yafeng Deng and Tianqiao Chen},
      year={2026},
      eprint={2603.23516},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.23516},
}
```

### Acknowledgments
This repository and documentation page are maintained by the MSA authors. For project updates, please visit the **Homepage**: https://evermind.ai/
