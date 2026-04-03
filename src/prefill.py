import sys
import os
from typing import List, Dict
from dataclasses import dataclass
import multiprocessing as mp
import pathlib
import torch

from transformers import AutoTokenizer, BitsAndBytesConfig

project_path = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_path))

from src.utils.gpu_worker import GpuWorker
from src.utils.template import QWEN3_TEMPLATE, QWEN3_INSTRUCT_TEMPLATE
from src.utils.cache import  copy_kv_cache_to_device, CustomDynamicCacheOnCPU
from src.msa.model import MSAForCausalLM
from src.utils.gpu_worker import GpuWorker
from src.utils.tools import compose_input
from src.types import Document, ProtocolConstants

@dataclass
class BlockModelInput:
    doc_input_ids: torch.Tensor
    doc_attention_mask: torch.Tensor
    doc_ids: torch.Tensor
    position_ids: torch.Tensor
    num_chunks: int
    chunk_sizes: List[int]

PREFILL_WORKER_READY = "PREFILL_WORKER_READY"
PREFILL_WORKER_CLOSE = "PREFILL_WORKER_CLOSE"
PREFILL_WORKER_MEMORY_DOCS = "PREFILL_WORKER_MEMORY_DOCS"
PREFILL_WORKER_NUM_CHUNKS_REPORT = "PREFILL_WORKER_NUM_CHUNKS_REPORT"
PREFILL_WORKER_META = "PREFILL_WORKER_META"

class PrefillStage1Worker(GpuWorker):
    """Memory工作进程"""

    def __init__(self, gpu_id: int, model_path: str,  template: dict,
                 pooling_kernel_size: int, envs: dict):
        """
        该 worker 被MemoryWorker创建并仅执行prefill stage 1获取block 的kv cache
        """
        super().__init__(gpu_id, envs)
        self.model_path = model_path
        self.pooling_kernel_size = pooling_kernel_size

        self._load_model()
        self.model_config = self.model.config

        self.template_id = -2
        self._prepare_template(template)

    def num_model_layers(self):
        return self.model_config.num_hidden_layers

    def _load_model(self):
        """加载模型和tokenizer"""
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # 加载模型
        self.model = MSAForCausalLM.from_pretrained(
            self.model_path,
            use_cache=True,
            attn_implementation="flash_attention_2",
            torch_dtype="bfloat16",
            device_map=self.device,
        )
        self.model.eval()

    
    @staticmethod
    def split_docs(docs: List[Document], block_size: int):
        sub_blocks = []
        curr_block = []
        sz = 0
        for doc in docs:
            chunks = doc.num_chunks
            if sz + chunks > block_size and curr_block:
                sub_blocks.append(curr_block)
                sz = 0
                curr_block = []
            curr_block.append(doc)
            sz += chunks
        
        if curr_block:
            sub_blocks.append(curr_block)
        
        return sub_blocks
    
    @staticmethod
    def wait_for_ready(q: mp.Queue):               
        ProtocolConstants.expect(q, PREFILL_WORKER_READY)

    @staticmethod
    def close_worker(q: mp.Queue):               
        ProtocolConstants.send(q, PREFILL_WORKER_CLOSE, block=True)

    @staticmethod
    def send_documents(q: mp.Queue, docs):
        ProtocolConstants.send(q,
                               PREFILL_WORKER_MEMORY_DOCS,
                               data=docs,
                               block=False)
    @staticmethod
    def recv_meta(q: mp.Queue):
        return ProtocolConstants.expect(q, PREFILL_WORKER_META)

    @staticmethod
    def prefill_worker_main(gpu_id: int, request_queue: mp.Queue, response_queue: mp.Queue,
                            model_path: str, template: Dict,
                            pooling_kernel_size: int, block_size: int, envs):
        
        # print(f"prefill worker {gpu_id} started")
        worker = PrefillStage1Worker(gpu_id, model_path, template, pooling_kernel_size, envs)

        # notify parent I'm ready
        ProtocolConstants.send(response_queue, PREFILL_WORKER_READY, block=False)

        docs: List[Document] = ProtocolConstants.expect(request_queue, PREFILL_WORKER_MEMORY_DOCS)
        try:
            for block in PrefillStage1Worker.split_docs(docs, block_size):
                meta = worker.inference(block)

                # send to master worker process and continue, DO NOT block
                ProtocolConstants.send(response_queue, 
                                    PREFILL_WORKER_META,
                                    data=meta,
                                    block=False)

        except Exception as e:
            print(f"[子进程 {gpu_id}] 发生错误: {e}")
            import traceback
            traceback.print_exc()
        
        # wait for exit signal
        ProtocolConstants.expect(request_queue, PREFILL_WORKER_CLOSE)
        print(f"prefill worker {gpu_id} ended")

    def inference(self, block: List[Document]):
        """
        处理memory block

        Args:
            memory_block: 分配给此GPU的memory block

        """
        model_input = self._prepare_block_inputs(block)

        kv_meta = self._inference(model_input)
        kv_meta['nr_docs'] = len(block)
        kv_meta['doc_ids'] = [item.doc_id for item in block]
        kv_meta['nr_chunks'] = [item.num_chunks for item in block]

        return kv_meta

    def _prepare_template(self, template: Dict) -> Dict:
        """
        重新加载单个memory block
        完整复制eval_anything_v2_batch.py中reload_memory的逻辑

        Args:
            block: memory block数据 [(doc_id, doc_str), ...]
            template: 模板字典

        Returns:
            KV cache元数据
        """
        # 获取模板信息
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.doc_end_id = self.tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"]

        prompt_template = template["prompt"].replace("{prompt}", self.pad_token)
        prompt_template_inputs = self.tokenizer(prompt_template, add_special_tokens=False)
        self.prompt_template_input_ids = prompt_template_inputs["input_ids"]
        self.prompt_template_attention_mask = prompt_template_inputs["attention_mask"]

        self.pad_index = self.prompt_template_input_ids.index(self.pad_token_id)
        self.template_head_input_ids = self.prompt_template_input_ids[:self.pad_index]
        self.template_head_attention_mask = self.prompt_template_attention_mask[:self.pad_index]
        self.template_tail_input_ids = self.prompt_template_input_ids[self.pad_index+1:]
        self.template_tail_attention_mask = self.prompt_template_attention_mask[self.pad_index+1:]


    def _prepare_block_inputs(self, block: List[Document]) -> Dict:
        """
        重新加载单个memory block
        完整复制eval_anything_v2_batch.py中reload_memory的逻辑

        Args:
            block: memory block数据 [(doc_id, doc_str), ...]
            template: 模板字典

        Returns:
            KV cache元数据
        """

        # 准备文档数据

        # block starts with template head
        doc_ids = [self.template_id] * len(self.template_head_input_ids)
        doc_input_ids = [i for i in self.template_head_input_ids ]
        doc_attention_mask = [i for i in self.template_head_attention_mask]
        position_ids = [i for i in range(self.pad_index)]

        chunk_sizes = [] # 记录每一份文档占用的 chunk 数量

        for doc_idx, item in enumerate(block):
            doc_id, doc, pre_calculated_num_chunk = item.doc_id, item.doc, item.num_chunks
            new_doc, doc_inputs = compose_input(doc, doc_id, self.tokenizer)
            # print(f"inference {doc_id+1}: {new_doc}")
            # 此处必须使用 1 起始的doc_idx，因为此 id 是用于生成 pool doc ID 的
            # 注意不可以使用doc_id，doc_id只能用于嵌入在语料中，使得 generate 时能生成出来，
            # 而pool doc ID的作用却是用于标注产生的 kv cache chunks 和文档的对应关系
            # 所以每次 stage1 的推理doc id 都是从 1 开始的
            temp_doc_ids = [doc_idx+1] * len(doc_inputs["input_ids"])
            temp_doc_input_ids = doc_inputs["input_ids"]
            temp_doc_attention_mask = doc_inputs["attention_mask"]
            length = len(temp_doc_input_ids)
            temp_position_ids = [i for i in range(length)]

            chunk_size = (len(temp_doc_ids) + self.pooling_kernel_size - 1) // self.pooling_kernel_size
            chunk_sizes.append(chunk_size)
            assert chunk_size == pre_calculated_num_chunk, f"pre calculated chunk {pre_calculated_num_chunk} got {chunk_size}, doc str {len(doc)} id len {length}/{len(temp_doc_ids)}: [{doc_id}] <{doc}>"



            doc_ids.extend(temp_doc_ids)
            doc_input_ids.extend(temp_doc_input_ids)
            doc_attention_mask.extend(temp_doc_attention_mask)
            position_ids.extend(temp_position_ids)

        input_ids_tensor = torch.LongTensor([doc_input_ids])
        attention_mask_tensor = torch.LongTensor([doc_attention_mask])
        doc_ids_tensor = torch.LongTensor([doc_ids])
        position_ids_tensor = torch.LongTensor([position_ids])

        return BlockModelInput(doc_input_ids=input_ids_tensor,
                               doc_attention_mask=attention_mask_tensor,
                               doc_ids=doc_ids_tensor,
                               position_ids=position_ids_tensor,
                               num_chunks=sum(chunk_sizes),
                               chunk_sizes=chunk_sizes)

    def _inference(self, model_input: BlockModelInput) -> Dict:
        """
        重新加载单个memory block
        完整复制eval_anything_v2_batch.py中reload_memory的逻辑

        Args:
            block: memory block数据 [(doc_id, doc_str), ...]
            template: 模板字典

        Returns:
            KV cache元数据
        """

        # 转换为tensor
        input_ids_tensor = model_input.doc_input_ids.to(self.device)
        attention_mask_tensor = model_input.doc_attention_mask.to(self.device)
        doc_ids_tensor = model_input.doc_ids.to(self.device)
        position_ids_tensor = model_input.position_ids.to(self.device)

        # 创建past_key_values，这里我们会直接将 kvcache 和其他的 tensor 全部保存在 cpu 上
        past_key_values = CustomDynamicCacheOnCPU()
        for layer_idx in range(self.num_model_layers()):
            past_key_values.record_kwargs(layer_idx, {"stage": "prefill_stage1"})

        # 执行prefill
        # TODO: 多 batch 会更快
        with torch.no_grad():
            if True:
                """我们的数据太大了，会导致model lm_head产生大量的显存堆积，所以直接用model.model避过去"""
                outputs = self.model.model(
                    input_ids=input_ids_tensor,
                    attention_mask=attention_mask_tensor,
                    position_ids=position_ids_tensor,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    output_docs_score=False,
                    doc_ids=doc_ids_tensor,
                )
            else:
                outputs = self.model(
                    input_ids=input_ids_tensor,
                    attention_mask=attention_mask_tensor,
                    doc_ids=doc_ids_tensor,
                    use_cache=True,
                    position_ids=position_ids_tensor,
                    past_key_values=past_key_values,
                )

        torch.cuda.empty_cache()
        # 构建返回的元数据
        kvcache_meta = {
            "chunk_sizes": model_input.chunk_sizes,
            "past_key_values": outputs.past_key_values,
        }

        return kvcache_meta

