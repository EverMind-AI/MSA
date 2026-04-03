from typing import Union, Dict, List
from dataclasses import dataclass

from src.utils.template import QWEN3_TEMPLATE, QWEN3_INSTRUCT_TEMPLATE

@dataclass
class GenerateConfig:
    devices: List[int] = None
    template: Union[str, Dict] = None
    max_generate_tokens: int = 256
    max_seq_len: int = 0 # total sequence length in a batch
    max_query_seq_len: int = 0 # max sequence for a single query
    max_batch_size: int = 0 # 0 if batch size is not limited
    top_p: float = 0.9
    temperature: float = 0.0
    qa_mode: bool = False

    def __post_init__(self):
        if isinstance(self.template, str):
            assert self.template in ["QWEN3_TEMPLATE", "QWEN3_INSTRUCT_TEMPLATE"]
            self.template = eval(self.template)
        assert isinstance(self.template, dict)

    @property
    def world(self):
        return len(self.devices) if self.devices else 0

@dataclass
class ModelConfig:
    model_path: str = "EverMind-AI/MSA-4B"

    doc_top_k: int = 16
    pooling_kernel_size: int = 64
    router_layer_idx: str = "all"

    # template 
    template_token_id = -2
    template_id_num = 3

    def get_model_envs(self):
        envs = {}
        # envs["TOP_K_DOCS"] = str(self.doc_top_k)
        # envs["POOLING_KERNEL_SIZE"] = str(self.pooling_kernel_size)
        # envs["ROUTER_LAYER_IDX"] = self.router_layer_idx
        return envs


@dataclass
class MemoryConfig:
    block_size: int = 16000  # 当对 memory 进行推理时使用的分块大小（tokens）
    slice_chunk_size: int = 16 * 1024
    pooling_kernel_size: int = 64
    memory_file_path: str = ""

    socket_ip: str = ""
    socket_port: int = 0