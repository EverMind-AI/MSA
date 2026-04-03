import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3Config,
    apply_rotary_pos_emb,
    repeat_kv,
)
try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    print("请安装flash-attn库: pip install flash-attn --no-build-isolation")
    flash_attn_varlen_func = None


class MemorySparseAttention(Qwen3Attention):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        if flash_attn_varlen_func is None:
            raise ImportError("flash_attn is required. Please install it via 'pip install flash-attn --no-build-isolation'")
        
        self.layer_idx = layer_idx
        self.top_k_docs = config.msa_config.top_k_docs
        self.pooling_kernel_size = config.msa_config.pooling_kernel_size
        self.router_layer_idx = config.msa_config.router_layer_idx

        if self.router_layer_idx == "all":
            self.router_layer_idx = list(range(config.num_hidden_layers))
        else:
            self.router_layer_idx = [int(i) for i in self.router_layer_idx.split(",")]
        self.is_router_layer = self.layer_idx in self.router_layer_idx

        self.head_reduce_method = config.msa_config.head_reduce_method
        self.query_reduce_method = config.msa_config.query_reduce_method
        self.chunk_reduce_method = config.msa_config.chunk_reduce_method
        self.decouple_pooling_mode = config.msa_config.decouple_pooling_mode
        self.aux_loss_method = config.msa_config.aux_loss_method

        self.decouple_router = config.msa_config.decouple_router
        if self.is_router_layer and self.decouple_router:
            self.router_k_proj = nn.Sequential(
                nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False),
                # nn.GELU(),
                # nn.Linear(config.num_key_value_heads * self.head_dim, config.num_key_value_heads * self.head_dim, bias=False)
            )
            self.router_q_proj = nn.Sequential(
                nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False),
                # nn.GELU(),
                # nn.Linear(config.num_attention_heads * self.head_dim, config.num_attention_heads * self.head_dim, bias=False)
            )
        self.num_kv_heads = config.num_key_value_heads

        self.sliding_window = None
        self.selected_docs_indices = None
        self.max_doc_id = None
        self.num_split_for_kv = 8
        self.template_prefix_kcache = None
        self.template_prefix_vcache = None
        self.memory_client = None

    def set_memory_client(self, memory_client):
        self.memory_client = memory_client

    def forward(
        self,
        hidden_states: torch.Tensor,
        doc_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        if self.training:
            return self._forward(
                hidden_states,
                doc_ids,
                attention_mask,
                position_embeddings,
                past_key_value,
                **kwargs,
            )
        elif past_key_value is not None:
            return self.forward_with_kvcache_for_batch_parrallel(
                hidden_states,
                doc_ids,
                attention_mask,
                position_embeddings,
                past_key_value,
                **kwargs,
            )
        else:
            raise Exception("error!")

    @staticmethod
    def map_tensor_to_group_ids(a: torch.Tensor) -> torch.Tensor:
        if a.ndim != 1:
            raise ValueError("输入 Tensor a 必须是一维的。")

        diff_mask = torch.diff(a) != 0  # [L-1]
        id_increments = diff_mask.int()  # [L-1]
        group_indices_offset = torch.cumsum(id_increments, dim=0)  # [L-1]

        b = torch.cat((
            torch.tensor([0], device=a.device, dtype=a.dtype), 
            group_indices_offset
        )) + 1
        
        return b

    def forward_with_kvcache_for_batch_parrallel(
        self,
        hidden_states: torch.Tensor,
        doc_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        bsz, q_len, _ = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype
        hidden_shape = (bsz, q_len, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        stage = past_key_value.cache_kwargs[self.layer_idx]["stage"]

        if stage == "prefill_stage1":
            max_doc_id = int(doc_ids.max().item())
            doc_token_mask = (doc_ids > 0) & (attention_mask == 1)
            doc_indices = torch.nonzero(doc_token_mask, as_tuple=False)
            original_doc_ids = doc_ids[doc_token_mask]
            original_doc_batch_indices = doc_indices[:, 0]
            global_doc_ids = original_doc_batch_indices * (max_doc_id + 1) + original_doc_ids
            
            if self.is_router_layer:
                _, counts = torch.unique_consecutive(global_doc_ids, return_counts=True)
                total_doc_tokens = global_doc_ids.shape[0]

                cu_seqlens = counts.cumsum(0)
                offsets = torch.zeros(counts.shape[0] + 1, dtype=counts.dtype, device=device)
                offsets[1:] = cu_seqlens
                offsets = offsets[:-1]

                expanded_offsets = torch.repeat_interleave(offsets, counts)
                original_order_ranks = torch.arange(total_doc_tokens, device=device) - expanded_offsets
                
                chunk_indices = original_order_ranks // self.pooling_kernel_size
                max_chunks_per_doc = (q_len // self.pooling_kernel_size) + 1

                global_chunk_ids = global_doc_ids * max_chunks_per_doc + chunk_indices

                unique_global_chunk_ids, chunk_token_counts = torch.unique_consecutive(global_chunk_ids, return_counts=True)
                pooled_doc_ids = unique_global_chunk_ids // max_chunks_per_doc % (max_doc_id + 1)


                pooled_k_chunks, pooled_v_chunks = self.sequence_pooling_kv(
                    key_states,
                    value_states,
                    doc_indices,
                    global_chunk_ids,
                )

                pooled_k_chunks = pooled_k_chunks.transpose(0, 1).unsqueeze(0)
                pooled_v_chunks = pooled_v_chunks.transpose(0, 1).unsqueeze(0)

                pooled_router_k = None
                if self.decouple_router:
                    r_k_raw = self.router_k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    r_k_docs = r_k_raw[doc_indices[:, 0], :, doc_indices[:, 1]]
                    
                    _, chunk_lengths = torch.unique_consecutive(global_chunk_ids, return_counts=True)
                    chunk_counts_view = chunk_lengths.view(-1, 1, 1).to(dtype=torch.float32)
                    b_k, h_k, d_k = r_k_docs.shape
                    k_flat = r_k_docs.reshape(b_k, -1).to(dtype=torch.float32)
                    k_cumsum = F.pad(torch.cumsum(k_flat, dim=0), (0, 0, 1, 0))
                    chunk_cu_seqlens = F.pad(torch.cumsum(chunk_lengths, 0), (1, 0))
                    k_sums_flat = k_cumsum[chunk_cu_seqlens[1:]] - k_cumsum[chunk_cu_seqlens[:-1]]
                    pooled_router_k = (k_sums_flat.view(unique_global_chunk_ids.shape[0], h_k, d_k) / chunk_counts_view).to(dtype=r_k_docs.dtype)
                    
                    pooled_router_k = pooled_router_k.transpose(0, 1).unsqueeze(0)
                
                if self.aux_loss_method == "INFONCE":
                    router_k = pooled_router_k if pooled_router_k is not None else pooled_k_chunks
                    pooled_router_k = F.normalize(router_k, p=2, dim=-1)

            if past_key_value is not None:
                num_template_mask_prefix = (doc_ids == -2).sum()
                template_prefix_kcache = key_states[:, :, :num_template_mask_prefix]
                template_prefix_vcache = value_states[:, :, :num_template_mask_prefix]
                kwargs = {
                    "template_prefix_kcache": template_prefix_kcache,
                    "template_prefix_vcache": template_prefix_vcache,
                }
                if self.is_router_layer:
                    pooled_k_chunks, pooled_v_chunks = past_key_value.update(pooled_k_chunks, pooled_v_chunks, self.layer_idx)
                    kwargs2 = {
                        "doc_id_bias": doc_ids.shape[1],
                        "pooled_doc_ids": pooled_doc_ids,
                        "prefill_stage1_kvcache_size": pooled_k_chunks.shape[2],
                    }
                    if pooled_router_k is not None:
                        past_key_value.update_router_kcache(pooled_router_k, self.layer_idx)
                    kwargs.update(kwargs2)
                past_key_value.record_kwargs(self.layer_idx, kwargs)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            
            attn_output = torch.zeros((bsz, q_len, self.config.num_attention_heads * self.head_dim), device=device, dtype=dtype)
            indices_b = torch.nonzero(doc_token_mask, as_tuple=False)
            
            if indices_b.shape[0] > 0:
                q_b, k_b, v_b = query_states[indices_b[:, 0], :, indices_b[:, 1]], key_states[indices_b[:, 0], :, indices_b[:, 1]], value_states[indices_b[:, 0], :, indices_b[:, 1]]
                doc_ids_b = doc_ids[indices_b[:, 0], indices_b[:, 1]]
                batch_indices_b = indices_b[:, 0]
                global_doc_ids_b = batch_indices_b * (max_doc_id + 1) + doc_ids_b
                _, counts_b = torch.unique_consecutive(global_doc_ids_b, return_counts=True)
                cu_seqlens_b = F.pad(torch.cumsum(counts_b, dim=0, dtype=torch.int32), (1, 0))
                output_b_flat = flash_attn_varlen_func(q_b, k_b, v_b, cu_seqlens_q=cu_seqlens_b, cu_seqlens_k=cu_seqlens_b, max_seqlen_q=int(counts_b.max()), max_seqlen_k=int(counts_b.max()), dropout_p=self.attention_dropout if self.training else 0.0, causal=True).view(-1, self.config.num_attention_heads * self.head_dim)
                attn_output[indices_b[:, 0], indices_b[:, 1]] += output_b_flat
            
            template_mask = (doc_ids == -2) & (attention_mask == 1)
            template_indices = torch.nonzero(template_mask, as_tuple=False)
            if template_indices.shape[0] > 0:
                q_template = query_states.transpose(1, 2)[template_mask]
                k_template = key_states.transpose(1, 2)[template_mask]
                v_template = value_states.transpose(1, 2)[template_mask]
                template_counts_per_sample = torch.bincount(template_indices[:, 0], minlength=bsz)
                cu_seqlens_template = F.pad(torch.cumsum(template_counts_per_sample, dim=0, dtype=torch.int32), (1, 0))
                output_template_flat = flash_attn_varlen_func(q_template, k_template, v_template, cu_seqlens_q=cu_seqlens_template, cu_seqlens_k=cu_seqlens_template, max_seqlen_q=int(template_counts_per_sample.max()), max_seqlen_k=int(template_counts_per_sample.max()), dropout_p=0.0, causal=True).view(-1, self.config.num_attention_heads * self.head_dim)
                attn_output[template_mask] = output_template_flat

            return self.o_proj(attn_output), None

        elif stage == "prefill_stage2":
            cache_kwargs = past_key_value.cache_kwargs[self.layer_idx]
            if self.memory_client is not None:
                if self.template_prefix_kcache is None:
                    self.template_prefix_kcache , self.template_prefix_vcache  = self.memory_client.get_template_prefix_kvcaches(self.layer_idx)
                    if not self.template_prefix_kcache.is_cuda:
                        self.template_prefix_kcache = self.template_prefix_kcache.to(device)
                    if not self.template_prefix_vcache.is_cuda:
                        self.template_prefix_vcache = self.template_prefix_vcache.to(device)
                template_prefix_kcache = self.template_prefix_kcache
                template_prefix_vcache = self.template_prefix_vcache
            else:
                template_prefix_kcache = cache_kwargs["template_prefix_kcache"].to(device)
                template_prefix_vcache = cache_kwargs["template_prefix_vcache"].to(device)

            final_k_to_scatter, final_v_to_scatter = None, None

            if self.is_router_layer:
                routing_q_for_scoring = self.router_q_proj(hidden_states).view(hidden_shape).transpose(1, 2) if self.decouple_router else query_states
                if self.aux_loss_method == "INFONCE":
                    routing_q_for_scoring = F.normalize(routing_q_for_scoring, p=2, dim=-1)

                query_mask = ((doc_ids == 0) & (attention_mask == 1))
                res = self.memory_client.doc_query(routing_q_for_scoring, query_mask, self.layer_idx)
                final_k_to_scatter, final_v_to_scatter, final_scores, num_selected_chunks_per_sample, final_selected_doc_ids = res

                if past_key_value.meta.get("require_recall_topk", False):
                    recall_topk_list = []
                    for i in range(bsz):
                        recall_topk_list.append({
                            "topk_doc_ids": final_selected_doc_ids[i].cpu().detach().tolist(),
                            "score": final_scores[i].cpu().detach().tolist(),
                        })
                    cache_kwargs["recall_topk"] = recall_topk_list
            else:
                num_selected_chunks_per_sample = torch.zeros(bsz, dtype=torch.long, device=device)

            num_q_per_sample = attention_mask.sum(dim=1)
            template_len = template_prefix_kcache.shape[2]
            kv_lengths = template_len + num_selected_chunks_per_sample + num_q_per_sample

            cu_seqlens_q = F.pad(num_q_per_sample.cumsum(0, dtype=torch.int32), (1, 0))
            cu_seqlens_kv = F.pad(kv_lengths.cumsum(0, dtype=torch.int32), (1, 0))
            
            total_q_tokens = cu_seqlens_q[-1].item()
            total_kv_tokens = cu_seqlens_kv[-1].item()

            q_final = torch.empty((total_q_tokens, self.config.num_attention_heads, self.head_dim), device=device, dtype=dtype)
            k_final_unrepeated = torch.empty((self.config.num_key_value_heads, total_kv_tokens, self.head_dim), device=device, dtype=dtype)
            v_final_unrepeated = torch.empty((self.config.num_key_value_heads, total_kv_tokens, self.head_dim), device=device, dtype=dtype)

            valid_q_mask = (attention_mask == 1)
            q_final = query_states.permute(0, 2, 1, 3)[valid_q_mask]

            offset_start_sample = cu_seqlens_kv[:-1]
            offset_start_template = offset_start_sample
            offset_start_chunks = offset_start_sample + template_len
            offset_start_question = offset_start_chunks + num_selected_chunks_per_sample

            template_indices = torch.arange(template_len, device=device).unsqueeze(0) + offset_start_template.unsqueeze(1)
            source_k_template = template_prefix_kcache.expand(bsz, -1, -1, -1).permute(1, 0, 2, 3).reshape(self.config.num_key_value_heads, -1, self.head_dim)
            k_final_unrepeated[:, template_indices.flatten(), :] = source_k_template
            source_v_template = template_prefix_vcache.expand(bsz, -1, -1, -1).permute(1, 0, 2, 3).reshape(self.config.num_key_value_heads, -1, self.head_dim)
            v_final_unrepeated[:, template_indices.flatten(), :] = source_v_template
            
            if self.is_router_layer and final_k_to_scatter is not None and final_k_to_scatter.shape[1] > 0:
                batch_indices_for_chunks = torch.arange(bsz, device=device).repeat_interleave(num_selected_chunks_per_sample)
                is_start_of_sample = torch.cat([torch.tensor([True], device=device), batch_indices_for_chunks[1:] != batch_indices_for_chunks[:-1]])
                cumsum_ranks = torch.ones_like(batch_indices_for_chunks).cumsum(0)
                start_offsets = cumsum_ranks[is_start_of_sample].repeat_interleave(num_selected_chunks_per_sample)
                chunk_rank_in_sample = cumsum_ranks - start_offsets

                chunk_dest_indices = offset_start_chunks[batch_indices_for_chunks] + chunk_rank_in_sample
                
                k_final_unrepeated[:, chunk_dest_indices, :] = final_k_to_scatter
                if final_v_to_scatter.device == torch.device("cpu"):
                    final_v_to_scatter = final_v_to_scatter.to(device)
                v_final_unrepeated[:, chunk_dest_indices, :] = final_v_to_scatter

            batch_indices_for_q = torch.arange(bsz, device=device).repeat_interleave(num_q_per_sample)
            q_rank_in_sample = (torch.cumsum(valid_q_mask.int(), dim=1) - 1)[valid_q_mask]
            q_dest_indices = offset_start_question[batch_indices_for_q] + q_rank_in_sample
            
            k_final_unrepeated[:, q_dest_indices, :] = key_states.permute(1, 0, 2, 3).reshape(self.config.num_key_value_heads, -1, self.head_dim)[:, valid_q_mask.flatten(), :]
            v_final_unrepeated[:, q_dest_indices, :] = value_states.permute(1, 0, 2, 3).reshape(self.config.num_key_value_heads, -1, self.head_dim)[:, valid_q_mask.flatten(), :]

            k_final = k_final_unrepeated
            v_final = v_final_unrepeated

            output_flat = flash_attn_varlen_func(
                q=q_final, k=k_final.transpose(0,1), v=v_final.transpose(0,1),
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_kv,
                max_seqlen_q=num_q_per_sample.max().item(), max_seqlen_k=kv_lengths.max().item(),
                dropout_p=0.0, causal=True
            ).view(-1, self.config.num_attention_heads * self.head_dim)

            attn_output = torch.zeros((bsz, q_len, self.config.num_attention_heads * self.head_dim), device=device, dtype=dtype)
            attn_output[valid_q_mask] = output_flat
            
            max_kv_len = kv_lengths.max().item()
            compacked_key_cache = torch.zeros((bsz, self.config.num_key_value_heads, max_kv_len, self.head_dim), dtype=dtype, device=device)
            compacked_value_cache = torch.zeros((bsz, self.config.num_key_value_heads, max_kv_len, self.head_dim), dtype=dtype, device=device)

            left_pad_mask = torch.arange(max_kv_len, device=device).unsqueeze(0) >= (max_kv_len - kv_lengths.unsqueeze(1))
            
            compacked_key_cache.permute(0, 2, 1, 3)[left_pad_mask] = k_final_unrepeated.permute(1, 0, 2)
            compacked_value_cache.permute(0, 2, 1, 3)[left_pad_mask] = v_final_unrepeated.permute(1, 0, 2)

            cache_kwargs["compacked_key_cache"] = compacked_key_cache
            cache_kwargs["compacked_value_cache"] = compacked_value_cache
            cache_kwargs["kv_lengths"] = kv_lengths
            cache_kwargs["attention_mask"] = left_pad_mask
            past_key_value.record_kwargs(self.layer_idx, cache_kwargs)
            
            return self.o_proj(attn_output), None

        else: 
            cache_kwargs = past_key_value.cache_kwargs[self.layer_idx]
            if "compacked_key_cache" not in cache_kwargs:
                raise ValueError("批次化紧凑KV缓存未找到。Prefill stage 2 是否正确运行?")

            compacked_key_cache = cache_kwargs["compacked_key_cache"]
            compacked_value_cache = cache_kwargs["compacked_value_cache"]
            kv_lengths = cache_kwargs["kv_lengths"]
            layer_attention_mask = cache_kwargs["attention_mask"]
            
            max_kv_len = compacked_key_cache.shape[2]
            full_k_unrepeated = torch.cat([compacked_key_cache, key_states], dim=2)
            full_v_unrepeated = torch.cat([compacked_value_cache, value_states], dim=2)

            if past_key_value.meta.get("qa_mode", False):
                cur_layer_attention_mask = torch.LongTensor([[1] * q_len for _ in range(bsz)]).to(device)
                cur_layer_attention_mask = (cur_layer_attention_mask * attention_mask).type(layer_attention_mask.dtype)
                layer_attention_mask = torch.cat([layer_attention_mask, cur_layer_attention_mask], dim=1)
                attn_mask_4d = layer_attention_mask[:, None, None, :].expand(-1, self.config.num_attention_heads, 1, -1)
                cache_kwargs["attention_mask"] = layer_attention_mask
            else:
                new_kv_lengths = kv_lengths + 1
                max_new_kv_len = max_kv_len + 1
                attn_mask_2d = torch.arange(max_new_kv_len, device=device).unsqueeze(0) >= (max_new_kv_len - new_kv_lengths.unsqueeze(1))

                attn_mask_4d = attn_mask_2d[:, None, None, :].expand(-1, self.config.num_attention_heads, 1, -1)
                cache_kwargs["kv_lengths"] = new_kv_lengths

            key_states_gqa = repeat_kv(full_k_unrepeated, self.num_key_value_groups)
            value_states_gqa = repeat_kv(full_v_unrepeated, self.num_key_value_groups)
            
            attn_output = F.scaled_dot_product_attention(
                query_states, 
                key_states_gqa, 
                value_states_gqa, 
                attn_mask=attn_mask_4d, 
                dropout_p=0.0, 
                is_causal=False
            ).transpose(1, 2).reshape(bsz, q_len, -1)
            
            cache_kwargs["compacked_key_cache"] = full_k_unrepeated
            cache_kwargs["compacked_value_cache"] = full_v_unrepeated
            past_key_value.record_kwargs(self.layer_idx, cache_kwargs)
            
            return self.o_proj(attn_output), None

    def _calculate_routing_scores_adaptive(
        self,
        query_states: torch.Tensor,     # [B, H, Q_len, D]
        pooled_k_bched: torch.Tensor,   # [B, C, H, D]
        routing_query_mask: torch.Tensor, # [B, Q_len] - 1 for valid, 0 for pad
        chunk_mask: torch.Tensor,       # [B, C] - 1 for valid, 0 for pad
    ) -> torch.Tensor:
        bsz, num_heads, q_len, head_dim = query_states.shape
        _, max_chunks, _, _ = pooled_k_bched.shape
        dtype, device = query_states.dtype, query_states.device
        min_val = torch.finfo(dtype).min

        k_states_T = pooled_k_bched.permute(0, 2, 3, 1)

        current_scaling = 1.0 if self.decouple_router and "INFONCE" in self.aux_loss_method else self.scaling
        scores = torch.matmul(query_states, k_states_T) * current_scaling
        
        q_mask_expanded = routing_query_mask.view(bsz, 1, q_len, 1)
        k_mask_expanded = chunk_mask.view(bsz, 1, 1, max_chunks)
        
        final_mask = q_mask_expanded & k_mask_expanded
        scores.masked_fill_(~final_mask, min_val)

        if self.head_reduce_method == "max":
            scores = scores.max(dim=1).values
        elif self.head_reduce_method == "mean":
            scores = scores.mean(dim=1)
        else:
            raise NotImplementedError(f"Unsupported head reduce method: {self.head_reduce_method}")

        if self.query_reduce_method == "max":
            scores_final = scores.max(dim=1).values

        elif self.query_reduce_method == "mean":
            valid_mask = final_mask.squeeze(1) # [B, Q_len, C]

            scores_clean = torch.where(valid_mask, scores, torch.zeros_like(scores))
            sum_scores = scores_clean.sum(dim=1) # [B, C]
            counts = valid_mask.sum(dim=1).to(dtype).clamp(min=1.0)
            mean_scores = sum_scores / counts

            scores_final = torch.where(
                chunk_mask, 
                mean_scores, 
                torch.tensor(min_val, device=device, dtype=dtype)
            )

        elif self.query_reduce_method == "last":
            q_lens = routing_query_mask.sum(dim=1).long()
            last_indices = (q_lens - 1).clamp(min=0)

            gather_idx = last_indices.view(bsz, 1, 1).expand(-1, 1, max_chunks)
            scores_final = scores.gather(1, gather_idx).squeeze(1)
            scores_final.masked_fill_(~chunk_mask, min_val)
            
        else:
            raise NotImplementedError(f"Unsupported query reduce method: {self.query_reduce_method}")

        return scores_final

    def sequence_pooling_kv(self, key_states, value_states, doc_indices, global_chunk_ids):
        k_docs = key_states[doc_indices[:, 0], :, doc_indices[:, 1]]
        v_docs = value_states[doc_indices[:, 0], :, doc_indices[:, 1]]
        unique_global_chunk_ids, chunk_lengths = torch.unique_consecutive(global_chunk_ids, return_counts=True)
        
        num_unique_chunks = unique_global_chunk_ids.shape[0]
        chunk_counts_view = chunk_lengths.view(-1, 1, 1).to(dtype=torch.float32)

        def compute_pooled_states_via_cumsum(states, counts_view, lengths):
            b, h, d = states.shape
            states_flat = states.reshape(b, -1).to(dtype=torch.float32)
            states_cumsum = F.pad(torch.cumsum(states_flat, dim=0), (0, 0, 1, 0))
            chunk_cu_seqlens = F.pad(torch.cumsum(lengths, 0), (1, 0))
            state_sums_flat = states_cumsum[chunk_cu_seqlens[1:]] - states_cumsum[chunk_cu_seqlens[:-1]]
            state_sums = state_sums_flat.view(num_unique_chunks, h, d)
            return (state_sums / counts_view).to(dtype=states.dtype)

        pooled_k_chunks = compute_pooled_states_via_cumsum(k_docs, chunk_counts_view, chunk_lengths)
        pooled_v_chunks = compute_pooled_states_via_cumsum(v_docs, chunk_counts_view, chunk_lengths)
        return pooled_k_chunks, pooled_v_chunks

    def sequence_pooling_qkv(self, query_states, key_states, value_states, doc_indices, global_chunk_ids):
        q_docs = query_states[doc_indices[:, 0], :, doc_indices[:, 1]]
        k_docs = key_states[doc_indices[:, 0], :, doc_indices[:, 1]]
        v_docs = value_states[doc_indices[:, 0], :, doc_indices[:, 1]]
        unique_global_chunk_ids, chunk_lengths = torch.unique_consecutive(global_chunk_ids, return_counts=True)
        
        num_unique_chunks = unique_global_chunk_ids.shape[0]
        chunk_counts_view = chunk_lengths.view(-1, 1, 1).to(dtype=torch.float32)
        def compute_pooled_states_via_cumsum(states, counts_view, lengths):
            b, h, d = states.shape
            states_flat = states.reshape(b, -1).to(dtype=torch.float32)
            states_cumsum = F.pad(torch.cumsum(states_flat, dim=0), (0, 0, 1, 0))
            chunk_cu_seqlens = F.pad(torch.cumsum(lengths, 0), (1, 0))
            
            state_sums_flat = states_cumsum[chunk_cu_seqlens[1:]] - states_cumsum[chunk_cu_seqlens[:-1]]
            state_sums = state_sums_flat.view(num_unique_chunks, h, d)
            return (state_sums / counts_view).to(dtype=states.dtype)

        pooled_q_chunks = compute_pooled_states_via_cumsum(q_docs, chunk_counts_view, chunk_lengths)
        pooled_k_chunks = compute_pooled_states_via_cumsum(k_docs, chunk_counts_view, chunk_lengths)
        pooled_v_chunks = compute_pooled_states_via_cumsum(v_docs, chunk_counts_view, chunk_lengths)
        return pooled_q_chunks, pooled_k_chunks, pooled_v_chunks

    def count_chunks_per_batch(self, doc_ids, attention_mask, kernel_size):
        batch_size = doc_ids.size(0)
        chunk_counts = []

        for i in range(batch_size):
            mask = attention_mask[i]
            ids = doc_ids[i]
            valid_ids = ids[mask == 1]
            
            if len(valid_ids) == 0:
                chunk_counts.append(0)
                continue
            _, counts = torch.unique_consecutive(valid_ids, return_counts=True)
            
            num_chunks = (counts + kernel_size - 1) // kernel_size
            total_chunks = num_chunks.sum().item()
            chunk_counts.append(total_chunks)

        return torch.LongTensor(chunk_counts).to(doc_ids.device)

    def _forward(
        self,
        hidden_states: torch.Tensor,
        doc_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype
        hidden_shape = (bsz, q_len, -1, self.head_dim)
        
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        routing_query_mask = (doc_ids == 0) & (attention_mask == 1)
        doc_token_mask = (doc_ids > 0) & (attention_mask == 1)

        query_indices = torch.nonzero(routing_query_mask, as_tuple=False)
        doc_indices = torch.nonzero(doc_token_mask, as_tuple=False)
        
        if doc_indices.shape[0] == 0 or query_indices.shape[0] == 0:
            raise ValueError("No query or doc tokens found")
        
        max_doc_id = int(doc_ids.max().item())
        attn_output = torch.zeros((bsz, q_len, self.config.num_attention_heads * self.head_dim), device=device, dtype=dtype)
        if self.is_router_layer:
            original_doc_ids = doc_ids[doc_token_mask]
            original_doc_batch_indices = doc_indices[:, 0]
            
            global_doc_ids = original_doc_batch_indices * (max_doc_id + 1) + original_doc_ids
            _, counts = torch.unique_consecutive(global_doc_ids, return_counts=True)
            total_doc_tokens = global_doc_ids.shape[0]
            
            offsets = torch.zeros(counts.shape[0] + 1, dtype=counts.dtype, device=device)
            offsets[1:] = counts.cumsum(0)
            offsets = offsets[:-1]

            expanded_offsets = torch.repeat_interleave(offsets, counts)
            original_order_ranks = torch.arange(total_doc_tokens, device=device) - expanded_offsets
            
            chunk_indices = original_order_ranks // self.pooling_kernel_size
            max_chunks_per_doc = (q_len // self.pooling_kernel_size) + 1

            global_chunk_ids = global_doc_ids * max_chunks_per_doc + chunk_indices
            
            routing_q_states = None
            routing_pooled_k_chunks = None

            if self.decouple_router:
                routing_q_states = self.router_q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                if "INFONCE" in self.aux_loss_method:
                    routing_q_states = F.normalize(routing_q_states, p=2, dim=-1)


                r_k_raw = self.router_k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                r_k_raw = repeat_kv(r_k_raw, self.num_key_value_groups)
                r_k_docs = r_k_raw[doc_indices[:, 0], :, doc_indices[:, 1]]
                
                unique_global_chunk_ids = torch.unique_consecutive(global_chunk_ids)
                _, chunk_lengths = torch.unique_consecutive(global_chunk_ids, return_counts=True)
                
                chunk_counts_view = chunk_lengths.view(-1, 1, 1).to(dtype=torch.float32)
                b_k, h_k, d_k = r_k_docs.shape
                k_flat = r_k_docs.reshape(b_k, -1).to(dtype=torch.float32)
                k_cumsum = F.pad(torch.cumsum(k_flat, dim=0), (0, 0, 1, 0))
                chunk_cu_seqlens = F.pad(torch.cumsum(chunk_lengths, 0), (1, 0))
                k_sums_flat = k_cumsum[chunk_cu_seqlens[1:]] - k_cumsum[chunk_cu_seqlens[:-1]]
                routing_pooled_k_chunks = (k_sums_flat.view(unique_global_chunk_ids.shape[0], h_k, d_k) / chunk_counts_view).to(dtype=r_k_docs.dtype)
                if "INFONCE" in self.aux_loss_method:
                    routing_pooled_k_chunks = F.normalize(routing_pooled_k_chunks, p=2, dim=-1)

                pooled_q_chunks = query_states[doc_indices[:, 0], :, doc_indices[:, 1]]
                pooled_k_chunks = key_states[doc_indices[:, 0], :, doc_indices[:, 1]]
                pooled_v_chunks = value_states[doc_indices[:, 0], :, doc_indices[:, 1]]
                num_doc_tokens = pooled_q_chunks.shape[0]
                num_chunks = num_doc_tokens // self.pooling_kernel_size

                pooled_q_chunks = pooled_q_chunks.view(num_chunks, self.pooling_kernel_size, self.num_heads, self.head_dim).mean(dim=1)
                pooled_k_chunks = pooled_k_chunks.view(num_chunks, self.pooling_kernel_size, self.num_heads, self.head_dim).mean(dim=1)
                pooled_v_chunks = pooled_v_chunks.view(num_chunks, self.pooling_kernel_size, self.num_heads, self.head_dim).mean(dim=1)
                num_heads = self.config.num_attention_heads
                head_dim = self.head_dim
            else:
                pooled_q_chunks, pooled_k_chunks, pooled_v_chunks = self.sequence_pooling_qkv(
                    query_states, 
                    key_states, 
                    value_states, 
                    doc_indices,
                    global_chunk_ids,
                )
                num_heads = self.config.num_attention_heads
                head_dim = self.head_dim
                
                routing_q_states = query_states
                routing_pooled_k_chunks = pooled_k_chunks
                if "INFONCE" in self.aux_loss_method:
                    routing_q_states = F.normalize(routing_q_states, p=2, dim=-1)
                    routing_pooled_k_chunks = F.normalize(routing_pooled_k_chunks, p=2, dim=-1)
                
            unique_global_chunk_ids = torch.unique_consecutive(global_chunk_ids)
            num_unique_chunks = unique_global_chunk_ids.shape[0]
            chunks_per_sample = self.count_chunks_per_batch(doc_ids, doc_token_mask, kernel_size=self.pooling_kernel_size)

            max_chunks = chunks_per_sample.max().item()
            pooled_router_k_bched = torch.zeros((bsz, max_chunks, num_heads, self.head_dim), device=device, dtype=dtype)
            chunk_mask = torch.arange(max_chunks, device=device).unsqueeze(0) < chunks_per_sample.unsqueeze(1)
            pooled_router_k_bched[chunk_mask] = routing_pooled_k_chunks
            q_lens = routing_query_mask.sum(dim=1) # (B,)
            max_q_len = int(q_lens.max().item())

            if max_q_len == 0:
                max_q_len = 1 
            valid_q_flat = routing_q_states.transpose(1, 2)[routing_query_mask] # [Total_Valid_Q, H, D]

            compact_q_states_t = torch.zeros(
                bsz, max_q_len, self.config.num_attention_heads, self.head_dim, 
                device=device, dtype=dtype
            )
            
            idx_range = torch.arange(max_q_len, device=device).unsqueeze(0)
            mask_compact_q = idx_range < q_lens.unsqueeze(1)
            
            compact_q_states_t[mask_compact_q] = valid_q_flat
            compact_q_states = compact_q_states_t.transpose(1, 2)

            max_scores_per_chunk = self._calculate_routing_scores_adaptive(
                compact_q_states,         # (B, H, S, D)
                pooled_router_k_bched,       # (B, C, H, D)
                mask_compact_q,   # (B, S)
                chunk_mask            # (B, C)
            )
            
            pooled_global_doc_ids = unique_global_chunk_ids // max_chunks_per_doc
            pooled_doc_ids_in_sample = pooled_global_doc_ids % (max_doc_id + 1)
            chunk_to_doc_id_flat = pooled_doc_ids_in_sample # 形状: (total_chunks, )

            chunk_to_doc_id_bched = torch.full((bsz, max_chunks), 0, dtype=torch.long, device=device)
            chunk_to_doc_id_bched[chunk_mask] = chunk_to_doc_id_flat

            offsets = torch.arange(bsz, device=device) * (max_doc_id + 1)
            global_chunk_to_doc_id = chunk_to_doc_id_bched + offsets.unsqueeze(1)
            flat_doc_scores = torch.full((bsz * (max_doc_id + 1),), -float('inf'), device=device, dtype=dtype)

            valid_scores_flat = max_scores_per_chunk[chunk_mask]
            valid_global_doc_ids_flat = global_chunk_to_doc_id[chunk_mask]

            if self.chunk_reduce_method == "max":
                doc_scores = flat_doc_scores.scatter_reduce(
                    dim=0, 
                    index=valid_global_doc_ids_flat, 
                    src=valid_scores_flat, 
                    reduce="amax", 
                    include_self=True
                )
            
            elif self.chunk_reduce_method == "mean":
                flat_doc_sums = torch.zeros_like(flat_doc_scores)
                
                flat_doc_sums = flat_doc_sums.scatter_reduce(
                    dim=0,
                    index=valid_global_doc_ids_flat,
                    src=valid_scores_flat,
                    reduce="sum",
                    include_self=False 
                )
                
                flat_doc_counts = torch.zeros_like(flat_doc_scores)
                ones = torch.ones_like(valid_scores_flat)
                
                flat_doc_counts = flat_doc_counts.scatter_reduce(
                    dim=0,
                    index=valid_global_doc_ids_flat,
                    src=ones,
                    reduce="sum",
                    include_self=False
                )
                
                flat_doc_counts_safe = flat_doc_counts.clamp(min=1.0)
                mean_scores = flat_doc_sums / flat_doc_counts_safe
                
                doc_scores = torch.where(
                    flat_doc_counts > 0,
                    mean_scores,
                    flat_doc_scores # 这里是 -inf
                )

            else:
                raise ValueError(f"Invalid chunk reduction method: {self.chunk_reduce_method}")

            scores_by_batch = doc_scores.view(bsz, -1)
            return_scores_by_batch = scores_by_batch.clone()

            num_docs_per_sample = (scores_by_batch > -1e9).sum(dim=1)
            # 为每个样本计算k值：取配置的top_k和实际文档数的较小者
            k_per_sample = torch.min(num_docs_per_sample, torch.full_like(num_docs_per_sample, self.top_k_docs))

            _, sorted_indices = torch.sort(scores_by_batch, dim=1, descending=True)
            
            range_tensor = torch.arange(scores_by_batch.shape[1], device=device).expand(bsz, -1)
            selection_mask = range_tensor < k_per_sample.unsqueeze(1)
            
            selected_docs_indices = sorted_indices.masked_fill(~selection_mask, -50)
            
            prompt_and_response_mask = (doc_ids < 1) & (attention_mask == 1)
            # 此处的 selected_docs_indices 已经是修复后的张量，所以这行代码无需修改
            selected_docs_mask = torch.any(doc_ids.unsqueeze(-1) == selected_docs_indices.unsqueeze(1), dim=-1) & doc_token_mask

            pa_indices = torch.nonzero(prompt_and_response_mask, as_tuple=False)
            q_pa_flat = query_states[pa_indices[:, 0], :, pa_indices[:, 1]]
            k_pa_flat = key_states[pa_indices[:, 0], :, pa_indices[:, 1]]
            v_pa_flat = value_states[pa_indices[:, 0], :, pa_indices[:, 1]]
            sort_key_pa = pa_indices[:, 0] * q_len + pa_indices[:, 1]
            
            selected_doc_token_indices = torch.nonzero(selected_docs_mask, as_tuple=False)
            is_doc_token_mask_flat = doc_token_mask.flatten()
            global_chunk_ids_padded = torch.full((bsz * q_len,), -1, dtype=torch.long, device=device)
            global_chunk_ids_padded[is_doc_token_mask_flat] = global_chunk_ids
            selected_chunk_ids_flat = global_chunk_ids_padded.view(bsz, q_len)[selected_docs_mask]
            
            unique_selected_chunk_ids, inverse_indices_fix = torch.unique(selected_chunk_ids_flat, sorted=True, return_inverse=True)
            if unique_selected_chunk_ids.numel() > 0:
                first_occurrence_indices = torch.empty_like(unique_selected_chunk_ids, dtype=torch.long)
                first_occurrence_indices.scatter_reduce_(src=torch.arange(selected_chunk_ids_flat.numel(), device=device),index=inverse_indices_fix, dim=0, reduce='amin', include_self=False)
                
                representative_indices = selected_doc_token_indices[first_occurrence_indices]
                sort_key_chunks = representative_indices[:, 0] * q_len + representative_indices[:, 1]

                map_gcid_to_poolidx = torch.full((int(global_chunk_ids.max().item()) + 1,), -1, dtype=torch.long, device=device)
                map_gcid_to_poolidx[unique_global_chunk_ids] = torch.arange(num_unique_chunks, device=device)
                
                pool_indices_to_gather = map_gcid_to_poolidx[unique_selected_chunk_ids]
                
                assert (pool_indices_to_gather.sort().values != pool_indices_to_gather).sum() == 0
                q_pooled_sel_flat = pooled_q_chunks[pool_indices_to_gather]
                k_pooled_sel_flat = pooled_k_chunks[pool_indices_to_gather]
                v_pooled_sel_flat = pooled_v_chunks[pool_indices_to_gather]

                batch_indices_chunks = representative_indices[:, 0]
            else:
                sort_key_chunks = torch.tensor([], dtype=torch.long, device=device)
                q_pooled_sel_flat = torch.tensor([], dtype=dtype, device=device).view(0, num_heads, head_dim)
                k_pooled_sel_flat = torch.tensor([], dtype=dtype, device=device).view(0, num_heads, head_dim)
                v_pooled_sel_flat = torch.tensor([], dtype=dtype, device=device).view(0, num_heads, head_dim)
                batch_indices_chunks = torch.tensor([], dtype=torch.long, device=device)

            q_combined = torch.cat([q_pa_flat, q_pooled_sel_flat], dim=0)
            k_combined = torch.cat([k_pa_flat, k_pooled_sel_flat], dim=0)
            v_combined = torch.cat([v_pa_flat, v_pooled_sel_flat], dim=0)
            
            combined_sort_keys = torch.cat([sort_key_pa, sort_key_chunks], dim=0)
            _, final_sort_indices = torch.sort(combined_sort_keys)
            
            q_a_final = q_combined[final_sort_indices]
            k_a_final = k_combined[final_sort_indices]
            v_a_final = v_combined[final_sort_indices]
            
            # 4.4 计算cu_seqlens (逻辑不变)
            batch_indices_pa = pa_indices[:, 0]
            batch_indices_combined = torch.cat([batch_indices_pa, batch_indices_chunks], dim=0)
            sorted_batch_indices = batch_indices_combined[final_sort_indices]
            
            batch_counts_a = torch.bincount(sorted_batch_indices, minlength=bsz)
            cu_seqlens_a = F.pad(torch.cumsum(batch_counts_a, dim=0, dtype=torch.int32), (1, 0))
        else:
            prompt_and_response_mask = (doc_ids < 1) & (attention_mask == 1)
            pa_indices = torch.nonzero(prompt_and_response_mask, as_tuple=False)
            q_a_final = query_states[pa_indices[:, 0], :, pa_indices[:, 1]]
            k_a_final = key_states[pa_indices[:, 0], :, pa_indices[:, 1]]
            v_a_final = value_states[pa_indices[:, 0], :, pa_indices[:, 1]]
            batch_counts_a = prompt_and_response_mask.sum(dim=1)
            cu_seqlens_a = F.pad(torch.cumsum(batch_counts_a, dim=0, dtype=torch.int32), (1, 0))
            return_scores_by_batch = None
        
        if q_a_final.shape[0] > 0:
            output_a_final = flash_attn_varlen_func(
                q_a_final, k_a_final, v_a_final,
                cu_seqlens_q=cu_seqlens_a, cu_seqlens_k=cu_seqlens_a,
                max_seqlen_q=int(batch_counts_a.max()), max_seqlen_k=int(batch_counts_a.max()),
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=True
            ).view(-1, self.config.num_attention_heads * self.head_dim)
            
            if self.is_router_layer:
                is_pa_mask_combined = torch.cat([
                    torch.ones(pa_indices.shape[0], dtype=torch.bool, device=device),
                    torch.zeros(q_pooled_sel_flat.shape[0], dtype=torch.bool, device=device) # 修正为使用池化块的数量
                ], dim=0)
                is_pa_mask_sorted = is_pa_mask_combined[final_sort_indices]
                
                output_pa_part = output_a_final[is_pa_mask_sorted]
                attn_output[pa_indices[:, 0], pa_indices[:, 1]] = output_pa_part
            else:
                attn_output[pa_indices[:, 0], pa_indices[:, 1]] = output_a_final

        indices_b = torch.nonzero(doc_token_mask, as_tuple=False)
        if indices_b.shape[0] > 0:
            q_b, k_b, v_b = query_states[indices_b[:, 0], :, indices_b[:, 1]], key_states[indices_b[:, 0], :, indices_b[:, 1]], value_states[indices_b[:, 0], :, indices_b[:, 1]]
            doc_ids_b = doc_ids[indices_b[:, 0], indices_b[:, 1]]
            batch_indices_b = indices_b[:, 0]
            
            global_doc_ids_b = batch_indices_b * (max_doc_id + 1) + doc_ids_b
            
            _, counts_b = torch.unique_consecutive(global_doc_ids_b, return_counts=True)
            cu_seqlens_b = F.pad(torch.cumsum(counts_b, dim=0, dtype=torch.int32), (1, 0))
            
            output_b_flat = flash_attn_varlen_func(
                q_b, k_b, v_b, cu_seqlens_q=cu_seqlens_b, cu_seqlens_k=cu_seqlens_b,
                max_seqlen_q=int(counts_b.max()), max_seqlen_k=int(counts_b.max()),
                dropout_p=self.attention_dropout if self.training else 0.0, causal=True
            ).view(-1, self.config.num_attention_heads * self.head_dim)
            
            attn_output[indices_b[:, 0], indices_b[:, 1]] += output_b_flat

        return (self.o_proj(attn_output), return_scores_by_batch), None