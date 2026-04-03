## Project Structure

```
MSA/
├── scripts/
│   ├── run_benchmarks.sh          # Run inference on benchmarks
│   ├── calculate_llm_score.sh     # LLM-based answer evaluation
│   └── resave_model.sh            # Convert base model to MSA format
└── src/
    ├── msa/                       # Core MSA implementation
    │   ├── configuration_msa.py   # MSA model configuration
    │   ├── memory_sparse_attention.py  # MemorySparseAttention layer
    │   ├── model.py               # MSAForCausalLM / MSAModel
    │   └── generate.py            # Generation logic
    ├── config/
    │   └── memory_config.py       # GenerateConfig, ModelConfig, MemoryConfig
    ├── evaluation/
    │   └── llm_judge.py           # LLM-based evaluation metrics
    ├── app/
    │   └── benchmark.py           # Benchmark runner
    ├── utils/                     # GPU workers, caching, templates, etc.
    ├── msa_service.py             # Multi-GPU inference engine (MSAEngine)
    ├── prefill.py                 # Stage 1 prefill worker
    ├── benchmarks.py              # Benchmark registry & specs
    └── types.py                   # Core type definitions
```

## Installation

**1. Create conda environment**

```bash
conda create -n msa python=3.12 -y
conda activate msa
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

<details>
<summary>requirements.txt</summary>

```
tqdm
openai
torch==2.6
torchvision==0.21.0
transformers==4.51.3
liger_kernel==0.5.10
accelerate==1.0.1
```

</details>

**3. Install Flash Attention**

Option A — build from source:

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Option B — install prebuilt wheel (CUDA 12, Python 3.12):

```bash
wget -P /tmp https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install /tmp/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
rm /tmp/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

## Download

**1. Download model**

```bash
mkdir ckpt
pip install -U huggingface_hub==0.31.4
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download EverMind-AI/MSA-4B --local-dir ckpt/MSA-4B
```

**2. Download benchmarks**

Benchmark data is hosted on [EverMind-AI/MSA-RAG-BENCHMARKS](https://huggingface.co/datasets/EverMind-AI/MSA-RAG-BENCHMARKS) and will be automatically downloaded to `data/` on first run, based on the benchmarks specified in `scripts/run_benchmarks.sh`. No manual download is needed.

## Quick Start

**1. Run inference on benchmarks**

```bash
bash scripts/run_benchmarks.sh eval_benchmark
```

**2. Compute LLM-based scores**

```bash
bash scripts/calculate_llm_score.sh eval_benchmark
```

## Supported Benchmarks

| Category | Benchmark |
|---|---|
| Multi-hop QA | `2wikimultihopqa`, `hotpotqa`, `musique` |
| Single-hop QA | `nature_questions`, `triviaqa_06M`, `triviaqa_10M`, `msmarco_v1`, `dureader`, `ms_100M`, `hipporag_narrative`, `hipporag_popqa` |
