import copy
from typing import List, Union, Dict

import PIL.Image
import torch
import numpy as np
import torchvision.transforms.functional as F
import transformers

from transformers import PreTrainedTokenizer

IGNORE_INDEX = -100

def print_trainable_params(model: torch.nn.Module) -> None:
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param))


def post_process_generate_ids(tokenizer: PreTrainedTokenizer, ids: torch.Tensor):
    ids = copy.deepcopy(ids)  # do not modify origin preds and targets
    ids[ids < 0] = tokenizer.pad_token_id
    # pad_to_multiof 开启后, 多余的部分没法解码, 这里暂时替换为 ','
    ids[ids >= len(tokenizer)] = tokenizer.convert_tokens_to_ids(',')
    return ids


def decode_generate_ids(tokenizer: PreTrainedTokenizer, ids: torch.Tensor) -> Union[List[str], str]:
    assert ids.ndim in [1, 2]
    only_one_sentence = ids.ndim == 1
    if only_one_sentence:
        ids = ids.unsqueeze(0)
    ids = post_process_generate_ids(tokenizer, ids)
    res = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    if only_one_sentence:
        return res[0]
    return res



def draw_bounding_boxes(
        image: Union[torch.Tensor, PIL.Image.Image],
        boxes: Union[torch.Tensor, List, np.ndarray],
        **kwargs,
):
    if isinstance(image, PIL.Image.Image):
        from torchvision.transforms import PILToTensor
        image = PILToTensor()(image)
    assert isinstance(image, torch.Tensor), ""

    if not isinstance(boxes, torch.Tensor):
        boxes = torch.as_tensor(boxes)
    assert isinstance(boxes, torch.Tensor)

    from torchvision.utils import draw_bounding_boxes as _draw_bounding_boxes
    return _draw_bounding_boxes(image, boxes, **kwargs)


# https://github.com/huggingface/tokenizers/issues/247#issuecomment-675458087
def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def patch_transformer_logging():
    import logging
    import transformers
    def enable_explicit_format():
        handlers = transformers.utils.logging._get_library_root_logger().handlers

        for handler in handlers:
            formatter = logging.Formatter("[(%(levelname)s) %(pathname)s:%(lineno)s ] %(asctime)s >> %(message)s")
            handler.setFormatter(formatter)
    transformers.utils.logging.enable_explicit_format = enable_explicit_format

def print_model_stats(model):
    """
    通用模型参数统计工具。
    自动识别是 Dense 还是 MoE 模型，并计算 Total vs Active 参数量。
    """
    
    # 1. 计算物理总参数量 (Total Parameters)
    # 使用 set 避免计算共享参数 (Shared Weights)，例如 Embedding 和 lm_head 共享权重的情况
    unique_params = {p.data_ptr(): p for p in model.parameters()}.values()
    total_params = sum(p.numel() for p in unique_params)
    
    # 2. 初始化激活参数量 (Active Parameters)
    # 默认假设是 Dense 模型，所有参数都是激活的
    active_params = total_params
    
    moe_infos = [] # 用于存储发现的 MoE 层信息
    
    # 3. 遍历所有子模块，寻找 MoE 层特征
    # 我们不匹配类名，而是匹配"特征" (Duck Typing)
    for name, module in model.named_modules():
        # 特征判定：有 num_experts 属性，且有一个叫 experts 的 ModuleList
        if hasattr(module, 'num_experts') and hasattr(module, 'experts') and isinstance(module.experts, nn.ModuleList):
            
            # 获取关键超参
            num_experts = getattr(module, 'num_experts', 0)
            # 兼容不同的 top_k 命名 (top_k 或 num_experts_per_tok)
            top_k = getattr(module, 'top_k', getattr(module, 'num_experts_per_tok', 0))
            
            # 如果找不到 top_k，可能不是标准的 Sparse MoE，跳过
            if top_k == 0: 
                continue

            # --- 核心计算逻辑 ---
            # 1. 计算单个专家的参数量 (假设所有专家结构相同，取第一个)
            # 这里必须用 recursion=True 确保统计专家内部所有层
            single_expert_params = sum(p.numel() for p in module.experts[0].parameters())
            
            # 2. 计算"休眠"专家数量
            dormant_experts = num_experts - top_k
            
            # 3. 从激活总数中扣除休眠专家的参数
            # 注意：total_params 里已经包含了 N 个专家，我们只需要减去 (N-K) 个
            if dormant_experts > 0:
                deduction = dormant_experts * single_expert_params
                active_params -= deduction
                
                moe_infos.append({
                    "layer": name,
                    "experts": num_experts,
                    "active": top_k,
                    "expert_size": single_expert_params
                })

    # --- 4. 格式化输出 ---
    def format_num(num):
        if num >= 1e9: return f"{num/1e9:.2f}B"
        if num >= 1e6: return f"{num/1e6:.2f}M"
        if num >= 1e3: return f"{num/1e3:.2f}K"
        return str(num)

    print("=" * 50)
    print(f"Model Architecture Analysis")
    print("=" * 50)
    
    if len(moe_infos) > 0:
        print(f"👉 Detection: MoE Model (Sparse Mixture-of-Experts)")
        print(f"   - Found {len(moe_infos)} MoE layers")
        print(f"   - Config: {moe_infos[0]['experts']} Experts, Top-{moe_infos[0]['active']} Active")
    else:
        print(f"👉 Detection: Dense Model (Standard Transformer)")
        
    print("-" * 50)
    print(f"Total Parameters (VRAM):  {format_num(total_params)}")
    print(f"Active Parameters (FLOPs): {format_num(active_params)}")
    
    if len(moe_infos) > 0:
        sparsity = 1 - (active_params / total_params)
        print(f"Sparsity Ratio:            {sparsity:.2%}")
        # 计算相比 Dense 版本的倍数
        # 假设 Dense 版本就是 active_params 大小（不太严谨但直观）
        print(f"Upcycling Scale:           {total_params/active_params:.2f}x Larger than Dense Base")
    else:
        print(f"Sparsity Ratio:            0.00% (Dense)")
        
    print("=" * 50)
    
    return total_params, active_params