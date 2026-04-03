#!/usr/bin/env python3
"""
缓存模块
包含CustomDynamicCache和CustomQuantizeDynamicCache两个类
从eval_anything_v2_batch.py中提取出来，实现模块化管理
"""

import copy
from typing import Optional, Tuple
import torch
from transformers.cache_utils import DynamicCache, QuantoQuantizedCache, QuantizedCacheConfig


class CustomDynamicCache(DynamicCache):
    """
    自定义动态缓存类
    扩展标准的DynamicCache，添加额外的元数据存储和查询功能
    """
    def __init__(self, _distributed_cache_data=None):
        super().__init__(_distributed_cache_data)
        self.cache_kwargs = {}
        self.group_cache = {}
        self.meta = {}
        self.router_key_cache = []

    def clear_kvcache(self):
        self.key_cache = []
        self.value_cache = []

    def record_kwargs(self, layer_idx, kwargs):
        """
        记录层的元数据信息

        Args:
            layer_idx: 层索引
            kwargs: 包含路由层信息的字典
        """
        if layer_idx in self.cache_kwargs:
            self.cache_kwargs[layer_idx].update(kwargs)
        else:
            self.cache_kwargs[layer_idx] = kwargs

    def get_layer_length(self):
        return len(self.cache_kwargs)

    def get_kvcache(self, layer_idx):
        """
        获取指定层的KV缓存

        Args:
            layer_idx: 层索引

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (key_cache, value_cache)
        """
        key_cache = self.key_cache[layer_idx]
        value_cache = self.value_cache[layer_idx]
        return key_cache, value_cache

    def get_router_kcache(self, layer_idx):
        if layer_idx < len(self.router_key_cache):
            return self.router_key_cache[layer_idx]
        else:
            return None

    def clear_query(self):
        """
        清理查询相关的临时数据
        移除查询过程中产生的临时数据，保持缓存清洁
        """
        for k, v in self.cache_kwargs.items():
            if "compacked_key_cache" in v:
                v.pop("compacked_key_cache")
                v.pop("compacked_value_cache")
            if "prefill_stage2_kvcache_size" in v:
                v.pop("prefill_stage2_kvcache_size")
            if "prefill_stage1_kvcache_size" in v:
                v.pop("prefill_stage1_kvcache_size")
            if "recall_topk" in v:
                v.pop("recall_topk")
        return self

    def get_seq_length(self, layer_idx=0) -> int:
        """
        返回缓存状态的序列长度

        Args:
            layer_idx: 可选的层索引

        Returns:
            int: 序列长度
        """
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or not self.key_cache[layer_idx].numel()  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def copy(self):
        """
        创建缓存的深拷贝

        Returns:
            CustomDynamicCache: 缓存的新副本
        """
        new_cache = CustomDynamicCache()
        new_cache.key_cache = [k.clone() for k in self.key_cache]
        new_cache.value_cache = [v.clone() for v in self.value_cache]
        new_cache.cache_kwargs = copy.deepcopy(self.cache_kwargs)
        new_cache.group_cache = copy.deepcopy(self.group_cache)
        new_cache.meta = copy.deepcopy(self.meta)
        new_cache._seen_tokens = self._seen_tokens
        return new_cache

    def update_router_kcache(
        self,
        key_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Update the cache
        if key_states is not None:
            if len(self.router_key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.router_key_cache), layer_idx):
                    self.router_key_cache.append(torch.tensor([]))
                self.router_key_cache.append(key_states)
            elif (
                not self.router_key_cache[layer_idx].numel()  # prefers not t.numel() to len(t) == 0 to export the model
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.router_key_cache[layer_idx] = key_states
            else:
                self.router_key_cache[layer_idx] = torch.cat([self.router_key_cache[layer_idx], key_states], dim=-2)

        return self.router_key_cache[layer_idx]
    
class CustomDynamicCacheOnCPU(CustomDynamicCache):
    def __init__(self, _distributed_cache_data=None):
        super().__init__(_distributed_cache_data)

    def record_kwargs(self, layer_idx, kwargs):
        d = {}
        for k, v in kwargs.items():
            if v is not None and torch.is_tensor(v):
                d[k] = v.cpu() if v.is_cuda else v.clone()
            else:
                d[k] = v
        super().record_kwargs(layer_idx, d)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        if key_states is not None and torch.is_tensor(key_states) and key_states.is_cuda:
            key_states = key_states.cpu()
        if value_states is not None and torch.is_tensor(value_states) and value_states.is_cuda:
            value_states = value_states.cpu()
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def update_router_kcache(
        self,
        key_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states is not None and torch.is_tensor(key_states) and key_states.is_cuda:
            key_states = key_states.cpu()
        return super().update_router_kcache(key_states, layer_idx)

class CustomQuantizeDynamicCache(QuantoQuantizedCache):
    """
    自定义量化动态缓存类
    扩展标准的QuantoQuantizedCache，添加额外的元数据存储和查询功能
    """
    def __init__(self, cache_config):
        super().__init__(cache_config)
        self.cache_config = cache_config
        self.cache_kwargs = {}
        self.group_cache = {}
        self.meta = {}

    def record_kwargs(self, layer_idx, kwargs):
        """
        记录层的元数据信息

        Args:
            layer_idx: 层索引
            kwargs: 包含路由层信息的字典
        """
        if layer_idx in self.cache_kwargs:
            self.cache_kwargs[layer_idx].update(kwargs)
        else:
            self.cache_kwargs[layer_idx] = kwargs

    def get_layer_length(self):
        return len(self.cache_kwargs)

    def clear_kvcache(self):
        self._quantized_key_cache = []
        self._quantized_value_cache = []
        self.key_cache = []
        self.value_cache = []

    def get_kvcache(self, layer_idx):
        """
        获取指定层的KV缓存（反量化后）

        Args:
            layer_idx: 层索引

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (key_cache, value_cache)
        """
        dequant_key = self._dequantize(self._quantized_key_cache[layer_idx])
        dequant_value = self._dequantize(self._quantized_value_cache[layer_idx])
        return dequant_key, dequant_value

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        更新缓存

        Args:
            key_states: 新的key状态
            value_states: 新的value状态
            layer_idx: 层索引
            cache_kwargs: 缓存关键字参数

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 更新后的key和value状态
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) < layer_idx:
            for i in range(len(self.key_cache), layer_idx):
                self.key_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
                self.value_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
                self._quantized_key_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
                self._quantized_value_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))

        if len(self.key_cache) == layer_idx:
            self._quantized_key_cache.append(self._quantize(key_states.contiguous(), axis=self.axis_key))
            self._quantized_value_cache.append(self._quantize(value_states.contiguous(), axis=self.axis_value))
            self.key_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
            self.value_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
            keys_to_return, values_to_return = key_states, value_states
        else:
            dequant_key = self._dequantize(self._quantized_key_cache[layer_idx])
            dequant_value = self._dequantize(self._quantized_value_cache[layer_idx])
            keys_to_return = [dequant_key, self.key_cache[layer_idx], key_states]
            values_to_return = [dequant_value, self.value_cache[layer_idx], value_states]

            keys_to_return = torch.cat(keys_to_return, dim=-2)
            values_to_return = torch.cat(values_to_return, dim=-2)
            if (
                self.key_cache[layer_idx].dim() == 4
                and self.key_cache[layer_idx].shape[-2] + 1 >= self.residual_length
            ):
                self._quantized_key_cache[layer_idx] = self._quantize(keys_to_return.contiguous(), axis=self.axis_key)
                self._quantized_value_cache[layer_idx] = self._quantize(
                    values_to_return.contiguous(), axis=self.axis_value
                )
                self.key_cache[layer_idx] = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
                self.value_cache[layer_idx] = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return keys_to_return, values_to_return

    def get_seq_length(self, layer_idx=0) -> int:
        """
        返回缓存状态的序列长度

        Args:
            layer_idx: 可选的层索引

        Returns:
            int: 序列长度
        """
        is_empty_layer = (
            len(self._quantized_key_cache) == 0  # no cache in any layer
            or len(self._quantized_key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or not self._quantized_key_cache[layer_idx].numel()  # the layer has no cache
        )
        layer_seq_length = self._quantized_key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def clear_query(self):
        """
        清理查询相关的临时数据
        移除查询过程中产生的临时数据，保持缓存清洁
        """
        for k, v in self.cache_kwargs.items():
            if "compacked_key_cache" in v:
                v.pop("compacked_key_cache")
                v.pop("compacked_value_cache")
            if "prefill_stage2_kvcache_size" in v:
                v.pop("prefill_stage2_kvcache_size")
            if "prefill_stage1_kvcache_size" in v:
                v.pop("prefill_stage1_kvcache_size")
            if "recall_topk" in v:
                v.pop("recall_topk")
        return self

    def copy(self):
        """
        创建缓存的深拷贝

        Returns:
            CustomQuantizeDynamicCache: 缓存的新副本
        """
        new_cache = CustomQuantizeDynamicCache(self.cache_config)
        if hasattr(self, '_quantized_key_cache'):
            new_cache._quantized_key_cache = [k.clone() for k in self._quantized_key_cache]
            new_cache._quantized_value_cache = [v.clone() for v in self._quantized_value_cache]
        new_cache.key_cache = [k.clone() for k in self.key_cache]
        new_cache.value_cache = [v.clone() for v in self.value_cache]
        new_cache.cache_kwargs = copy.deepcopy(self.cache_kwargs)
        new_cache.group_cache = copy.deepcopy(self.group_cache)
        new_cache.meta = copy.deepcopy(self.meta)
        new_cache._seen_tokens = self._seen_tokens
        return new_cache


def create_cache(quantize_nbits: Optional[int] = 0):
    """
    根据参数创建合适的缓存实例

    Args:
        args: 包含量化相关参数的命名空间对象

    Returns:
        CustomDynamicCache or CustomQuantizeDynamicCache: 缓存实例
    """
    if quantize_nbits > 0:
        quan_cache_config = QuantizedCacheConfig(nbits=quantize_nbits)
        return CustomQuantizeDynamicCache(quan_cache_config)
    else:
        return CustomDynamicCache()

def manual_deepcopy_kv_cache(cache_obj):
    """
    Manually performs a deep copy of a custom KV cache object,
    avoiding the issues with quanto's __deepcopy__.
    """
    # 1. 创建一个新的、空的 cache 对象实例
    if isinstance(cache_obj, CustomQuantizeDynamicCache):
        # 如果是量化缓存，需要传入配置
        new_cache = CustomQuantizeDynamicCache(cache_obj.cache_config)
    elif isinstance(cache_obj, CustomDynamicCache):
        new_cache = CustomDynamicCache()
    else:
        # 如果有其他类型的缓存，可以在这里扩展
        raise TypeError(f"Unsupported cache type for manual deepcopy: {type(cache_obj)}")

    # 2. 复制非张量元数据
    # 使用标准 deepcopy 是安全的，因为这些是字典和列表
    new_cache.cache_kwargs = copy.deepcopy(cache_obj.cache_kwargs)
    new_cache.meta = copy.deepcopy(cache_obj.meta)

    # 3. 逐层复制核心的 key-value 张量缓存
    if hasattr(cache_obj, '_quantized_key_cache'): # 处理 QuantoQuantizedCache
        for layer_cache in cache_obj._quantized_key_cache:
            # .clone().detach() 是安全复制张量的标准方法
            new_cache._quantized_key_cache.append(layer_cache.clone().detach())
        for layer_cache in cache_obj._quantized_value_cache:
            new_cache._quantized_value_cache.append(layer_cache.clone().detach())
        new_cache._seen_tokens = cache_obj._seen_tokens

    if hasattr(cache_obj, 'key_cache'): # 处理 DynamicCache
        for layer_cache in cache_obj.key_cache:
            new_cache.key_cache.append(layer_cache.clone().detach())
        for layer_cache in cache_obj.value_cache:
            new_cache.value_cache.append(layer_cache.clone().detach())
        new_cache._seen_tokens = cache_obj._seen_tokens

    return new_cache

def convert_tensor(data, cuda_device):
    """转换结构体的 tensor device，如果cuda_device非 None，则将cpu 转换到 cuda，否则将 cuda 转换到 cpu
    支持dict, list, tuple, set
    """
    
    converted_count = [0]  # 使用列表以便在嵌套函数中修改
    
    def _convert_recursive(obj):
        # 如果是torch tensor且在CUDA上
        if torch.is_tensor(obj):
            if cuda_device:
                return obj.to(cuda_device) if obj.is_cpu else obj
            if obj.is_cuda:
                return obj.cpu()
            return obj
        
        # 处理各种容器类型
        elif isinstance(obj, dict):
            return {k: _convert_recursive(v) for k, v in obj.items()}
        
        elif isinstance(obj, list):
            return [_convert_recursive(item) for item in obj]
        
        elif isinstance(obj, tuple):
            # 元组不可变，总是创建新的
            return tuple(_convert_recursive(item) for item in obj)
        
        elif isinstance(obj, set):
            return {_convert_recursive(item) for item in obj}
        
        # 其他数据类型直接返回
        else:
            return obj

    return _convert_recursive(data)

def copy_dict_to_cpu(d: dict):
    ret = {}
    for k, v in d.items():
        ret[k] = v.cpu() if torch.is_tensor(v) and v.is_cuda else v
    return ret

def copy_dict_to_gpu(d: dict, device):
    if not d:
        return d
    ret = {}
    for k, v in d.items():
        ret[k] = v.to(device) if torch.is_tensor(v) and not v.is_cuda else v
    return ret

def copy_kv_cache_to_device(cache_obj, cuda_device, copy_v: bool=True):
    if isinstance(cache_obj, CustomQuantizeDynamicCache):
        new_cache = CustomQuantizeDynamicCache(cache_obj.cache_config)
    elif isinstance(cache_obj, CustomDynamicCache):
        new_cache = CustomDynamicCache()
    else:
        raise TypeError(f"Unsupported cache type for manual deepcopy: {type(cache_obj)}")

    # 复制非张量元数据
    new_cache.cache_kwargs = convert_tensor(cache_obj.cache_kwargs, cuda_device)
    new_cache.meta = convert_tensor(cache_obj.meta, cuda_device)

    # 复制核心张量缓存
    if hasattr(cache_obj, '_quantized_key_cache'):
        new_cache._quantized_key_cache = convert_tensor(cache_obj._quantized_key_cache, cuda_device) 
        if copy_v:
            new_cache._quantized_value_cache = convert_tensor(cache_obj._quantized_value_cache, cuda_device) 
        else:
            new_cache._quantized_value_cache = cache_obj._quantized_value_cache
        new_cache._seen_tokens = cache_obj._seen_tokens

    if hasattr(cache_obj, 'key_cache'):
        new_cache.key_cache = convert_tensor(cache_obj.key_cache, cuda_device)
        if copy_v:
            new_cache.value_cache = convert_tensor(cache_obj.value_cache, cuda_device)
        else:
            new_cache.value_cache = cache_obj.value_cache
        new_cache._seen_tokens = cache_obj._seen_tokens

    return new_cache

# def copy_kv_cache_to_gpu(cache_obj, device, copy_v: bool=True):
#     """
#     手动执行缓存对象的深拷贝
#     避免quanto库__deepcopy__的问题

#     Args:
#         cache_obj: 要拷贝的缓存对象

#     Returns:
#         缓存对象的深拷贝副本
#     """
#     if isinstance(cache_obj, CustomQuantizeDynamicCache):
#         new_cache = CustomQuantizeDynamicCache(cache_obj.cache_config)
#     elif isinstance(cache_obj, CustomDynamicCache):
#         new_cache = CustomDynamicCache()
#     else:
#         raise TypeError(f"Unsupported cache type for manual deepcopy: {type(cache_obj)}")

#     # 复制非张量元数据
#     new_cache.cache_kwargs = copy_dict_to_gpu(cache_obj.cache_kwargs, device)
#     new_cache.meta = copy_dict_to_gpu(cache_obj.meta, device)

#     # 复制核心张量缓存
#     if hasattr(cache_obj, '_quantized_key_cache'):
#         new_cache._quantized_key_cache = [t.to(device) for t in cache_obj._quantized_key_cache]
#         if copy_v:
#             new_cache._quantized_value_cache = [t.to(device) for t in cache_obj._quantized_value_cache]
#         else:
#             new_cache._quantized_value_cache = cache_obj._quantized_value_cache
#         new_cache._seen_tokens = cache_obj._seen_tokens

#     if hasattr(cache_obj, 'key_cache'):
#         new_cache.key_cache = [t.to(device) for t in cache_obj.key_cache]
#         if copy_v:
#             new_cache.value_cache = [t.to(device) for t in cache_obj.value_cache]
#         else:
#             new_cache.value_cache = cache_obj.value_cache

#         new_cache._seen_tokens = cache_obj._seen_tokens

#     return new_cache