import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar

from huggingface_hub import hf_hub_download


class Category(Enum):
    """Benchmark categories with associated root directories and path patterns."""
    RAG = auto()
    RAG_0108 = auto()
    LENGTH_SCALE = auto()


# ============================================================================
# HuggingFace config & local data root
# ============================================================================

HF_REPO_ID = "EverMind-AI/MSA-RAG-BENCHMARKS"
_DATA_ROOT = os.path.join(os.getcwd(), "data")


@dataclass(frozen=True)
class BenchmarkSpec:
    """Immutable specification for a single benchmark's file layout."""
    bench_name: str       # benchmark name, also the HF subdirectory
    query_file: str
    memory_file: str

    def _resolve(self, filename: str) -> str:
        """Return local path if cached, otherwise download from HF into data/."""
        local_path = os.path.join(_DATA_ROOT, self.bench_name, filename)
        if os.path.exists(local_path):
            return local_path
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        return hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"{self.bench_name}/{filename}",
            repo_type="dataset",
            local_dir=_DATA_ROOT,
        )

    @property
    def query_path(self) -> str:
        return self._resolve(self.query_file)

    @property
    def memory_path(self) -> str:
        return self._resolve(self.memory_file)

    def get_bench_files(self) -> tuple[str, str]:
        return self.query_path, self.memory_path


# ============================================================================
# Registry: benchmark name -> spec
# ============================================================================

def _rag(name: str) -> BenchmarkSpec:
    return BenchmarkSpec(name, f"qdata_{name}.pkl", f"mdata_{name}.pkl")


def _rag_0108(name: str) -> BenchmarkSpec:
    return BenchmarkSpec(name, f"qdata_{name}.pkl", f"mdata_{name}.pkl")


_REGISTRY: dict[str, BenchmarkSpec] = {
    # --- Length-scale benchmarks ---
    "ms_100M": BenchmarkSpec("ms_100M", "qdata_msmarco_16K.pkl", "mdata_msmarco_100M.pkl"),
    # --- Multi-hop QA ---
    "2wikimultihopqa":    _rag("2wikimultihopqa"),
    "hotpotqa":           _rag("hotpotqa"),
    "musique":            _rag("musique"),
    # --- HippoRAG ---
    "hipporag_narrative": _rag_0108("hipporag_narrative"),
    "hipporag_popqa":     _rag_0108("hipporag_popqa"),
    # --- Single-hop QA ---
    "nature_questions":   _rag("nature_questions"),
    "triviaqa_06M":       _rag("triviaqa_06M"),
    "triviaqa_10M":       _rag("triviaqa_10M"),
    # --- Multilingual / Passage retrieval ---
    "dureader":           _rag("dureader"),
    "msmarco_v1":         _rag("msmarco_v1"),
}

ALL_BENCH_NAMES: list[str] = list(_REGISTRY)


# ============================================================================
# Public API
# ============================================================================

class BenchMarks:
    """Resolve benchmark name to query / memory file paths.

    Usage:
        bench = BenchMarks("hotpotqa")
        query_file, memory_file = bench.get_bench_files()
    """

    AVAILABLE: ClassVar[list[str]] = ALL_BENCH_NAMES

    def __init__(self, bench_name: str) -> None:
        if bench_name not in _REGISTRY:
            raise ValueError(
                f"Unknown benchmark: {bench_name!r}. "
                f"Available: {', '.join(ALL_BENCH_NAMES)}"
            )
        self._spec = _REGISTRY[bench_name]
        self.name = bench_name
        self.bench_name = self._spec.bench_name
        self.query_file_name = self._spec.query_file
        self.memory_file_name = self._spec.memory_file

    def get_bench_files(self) -> tuple[str, str]:
        return self._spec.get_bench_files()

    def __repr__(self) -> str:
        return f"BenchMarks({self.name!r})"
