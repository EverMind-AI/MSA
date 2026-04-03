import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, Tuple, List, Dict
from functools import partial
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Model, 
    FlashAttentionKwargs,
    Qwen3Config,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    DynamicCache,
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3Attention,
    Qwen3PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache
from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
from src.msa import MemorySparseAttention, MSAGenerationMixin, MSAConfig

@dataclass
class MSALayerModelOutputWithPast(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_docs_scores: Optional[Dict] = None

@dataclass
class MSACausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    answer_loss: Optional[torch.FloatTensor] = None
    reconstruction_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    temperature: Optional[torch.FloatTensor] = None
    train_router_metrics: Optional[Dict] = None

class MSADeocoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int, attn_type: str = "sparse_attention"):
        super().__init__(config=config, layer_idx=layer_idx)
        self.layer_idx = layer_idx
        self.attn_type = attn_type
        self.hidden_size = config.hidden_size
        if attn_type == "full_attention":
            self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        elif attn_type == "sparse_attention":
            self.self_attn = MemorySparseAttention(config=config, layer_idx=layer_idx)
        
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        config.sliding_window = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_docs_score: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        doc_ids: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        if self.attn_type == "full_attention":
            hidden_states, self_attn_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                doc_ids=doc_ids,
                input_ids=input_ids,
                **kwargs,
            )
        else:
            hidden_states, self_attn_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                doc_ids=doc_ids,
                input_ids=input_ids,
                **kwargs,
            )
        
        if isinstance(hidden_states, tuple):
            hidden_states, docs_score = hidden_states
        else:
            docs_score = None
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        
        if output_docs_score:
            outputs += (docs_score,)

        return outputs

class MSAModel(Qwen3Model):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rewrite_position = config.msa_config.rewrite_position
        self.layers = nn.ModuleList([
            MSADeocoderLayer(config, layer_idx, attn_type="sparse_attention") 
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_docs_score: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        doc_ids: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        
        if not self.rewrite_position and self.training:
            position_ids = None
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # causal_mask = self._update_causal_mask(
        #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        # )
        causal_mask = attention_mask

        hidden_states = inputs_embeds
        # import pdb;pdb.set_trace()
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_docs_scores = () if output_docs_score else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_docs_score,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    doc_ids,
                    input_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_docs_score=output_docs_score,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    doc_ids=doc_ids,
                    input_ids=input_ids,
                    **flash_attn_kwargs,
                )
            
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_docs_score:
                all_docs_scores += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MSALayerModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            all_docs_scores=all_docs_scores
        )
    
class MSAForCausalLM(Qwen3PreTrainedModel, MSAGenerationMixin):
    config_class = MSAConfig
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    
    def __init__(self, config):
        super().__init__(config)
        self.num_layers = config.num_hidden_layers
        self.router_layer_idx = config.msa_config.router_layer_idx

        if self.router_layer_idx == "all":
            self.router_layer_idx = list(range(config.num_hidden_layers))
        else:
            self.router_layer_idx = [int(i) for i in self.router_layer_idx.split(",")]

        self.mid_layers = config.num_hidden_layers // 2
        self.model = MSAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.aux_loss = config.msa_config.aux_loss
        self.lmloss_weigth = config.msa_config.lmloss_weigth
        self.auxloss_weight = config.msa_config.auxloss_weight
        self.recloss_weight = config.msa_config.recloss_weight
        self.ansloss_weight = config.msa_config.ansloss_weight
        self.aux_loss_method = config.msa_config.aux_loss_method  # INFONCE, BCE, INFONCE_DECOUPLE, INFONCE_DECOUPLE_FOCAL 
        self.decouple_router = config.msa_config.decouple_router
            
        if "INFONCE" in self.aux_loss_method:
            temperature = config.msa_config.infonce_loss_temp
            self.temperature = nn.Parameter(torch.ones([]) * temperature, requires_grad=False)
        elif self.aux_loss_method == "BCE":
            self.b = nn.Parameter(-20 * torch.ones([]), requires_grad=True)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_docs_score: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # msa
        doc_ids: Optional[torch.LongTensor] = None,
        batch_aux_labels: List[List[int]] = None,
        batch_reconstruction_labels: Optional[torch.LongTensor] = None,
        batch_answer_labels: Optional[torch.LongTensor] = None,
        train_qa_samples: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        output_docs_score = self.aux_loss
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_docs_score=output_docs_score,
            cache_position=cache_position,
            doc_ids=doc_ids,
            **kwargs,
        )

        hidden_states = outputs[0]
        
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        kept_hidden_states = hidden_states[:, slice_indices, :]

        shift_labels = kwargs.pop("shift_labels", None)
        logits = None
        loss = None
        reconstruction_loss = None
        aux_loss = None
        answer_loss = None
        # if in training mode, don't materialize logits
        if self.training and (labels is not None or shift_labels is not None):
            loss = LigerForCausalLMLoss(
                hidden_states=kept_hidden_states,
                lm_head_weight=self.lm_head.weight,
                labels=labels,
                shift_labels=shift_labels,
                hidden_size=self.config.hidden_size,
                **kwargs,
            )
            if batch_reconstruction_labels is not None:
                reconstruction_loss = LigerForCausalLMLoss(
                    hidden_states=kept_hidden_states,
                    lm_head_weight=self.lm_head.weight,
                    labels=batch_reconstruction_labels,
                    shift_labels=None,
                    hidden_size=self.config.hidden_size,
                    **kwargs,
                )
            else:
                reconstruction_loss = torch.tensor(0.0).to(hidden_states.device)

            if torch.sum(train_qa_samples) == 0:
                batch_answer_labels = None
            else:
                train_qa_samples_mask = train_qa_samples == 1
                temp_kept_hidden_states = kept_hidden_states[train_qa_samples_mask]
                temp_batch_answer_labels = batch_answer_labels[train_qa_samples_mask]

            if batch_answer_labels is not None:
                answer_loss = LigerForCausalLMLoss(
                    hidden_states=temp_kept_hidden_states,
                    lm_head_weight=self.lm_head.weight,
                    labels=temp_batch_answer_labels,
                    shift_labels=None,
                    hidden_size=self.config.hidden_size,
                    **kwargs,
                )
            else:
                answer_loss = torch.tensor(0.0).to(hidden_states.device)
            

        else:  # if in inference mode materialize logits
            logits = self.lm_head(kept_hidden_states)
            if labels is not None:
                loss = self.loss_function(
                    logits=logits,
                    labels=labels,
                    vocab_size=self.config.vocab_size,
                    **kwargs,
                )
                if batch_reconstruction_labels is not None:
                    reconstruction_loss = self.loss_function(
                        logits=logits,
                        labels=batch_reconstruction_labels,
                        vocab_size=self.config.vocab_size,
                        **kwargs,
                    )
                else:
                    reconstruction_loss = torch.tensor(0.0).to(hidden_states.device)

                if batch_answer_labels is not None:
                    answer_loss = self.loss_function(
                        logits=logits,
                        labels=batch_answer_labels,
                        vocab_size=self.config.vocab_size,
                        **kwargs,
                    )
                else:
                    answer_loss = torch.tensor(0.0).to(hidden_states.device)
            
        lm_loss = torch.tensor(0.0).to(hidden_states.device)
        if loss is not None:
            lm_loss = loss.clone()
        aux_loss = torch.tensor(0.0).to(hidden_states.device)
        train_router_metrics = None
        if batch_aux_labels is not None and self.aux_loss:
            aux_loss, train_router_metrics = self.caculate_aux_loss(aux_loss, outputs, batch_aux_labels, hidden_states.device, hidden_states.dtype)

        # Ensure all loss components have default values
        reconstruction_loss = reconstruction_loss if reconstruction_loss is not None else torch.tensor(0.0).to(hidden_states.device)
        answer_loss = answer_loss if answer_loss is not None else torch.tensor(0.0).to(hidden_states.device)
        aux_loss = aux_loss if aux_loss is not None else torch.tensor(0.0).to(hidden_states.device)
        
        if loss is not None:
            loss = self.lmloss_weigth * loss + \
                   self.recloss_weight * reconstruction_loss + \
                   self.auxloss_weight * aux_loss + \
                   self.ansloss_weight * answer_loss
                   
        return MSACausalLMOutputWithPast(
            loss=loss,
            lm_loss=lm_loss,
            aux_loss=aux_loss,
            answer_loss=answer_loss,
            reconstruction_loss=reconstruction_loss,
            train_router_metrics=train_router_metrics,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            temperature=self.temperature if "INFONCE" in self.aux_loss_method else torch.tensor(0.0).to(hidden_states.device),
        )

    def calculate_decoupled_infonce_loss(self, logits, label, num_pos):
        """
        改进版：Decoupled InfoNCE
        解决了多正样本之间的互斥问题，检索任务推荐使用。
        """
        with torch.no_grad():
            temperature = self.temperature.clamp(0.001, 0.5)
        
        # 1. 缩放
        scaled_logits = logits / temperature
        
        # 2. 数值稳定：转为 exp 域
        # 减去最大值防止溢出 (标准 Softmax trick)
        max_logits = torch.max(scaled_logits, dim=0, keepdim=True)[0].detach()
        exp_logits = torch.exp(scaled_logits - max_logits)
        
        # 获取所有负样本的 exp 之和
        neg_exp_sum = torch.sum(exp_logits * (1 - label), dim=0, keepdim=True)
        
        # 4. 计算 Log Prob
        # 对于每一个正样本 i:
        # Prob_i = exp_i / (exp_i + neg_exp_sum)
        # 这样分母里就没有“其他正样本”在竞争了
        denominators = exp_logits + neg_exp_sum
        
        log_probs = scaled_logits - max_logits - torch.log(denominators + 1e-10)
        
        # 5. 计算 Loss
        # 只取正样本位置的 log_prob
        loss_map = - log_probs * (label / num_pos.clamp(min=1.0))
        
        return loss_map.sum(dim=0)

    def caculate_infonce_loss(self, logits, label, num_pos):
        # 限制温度范围避免数值不稳定
        with torch.no_grad():
            temperature = self.temperature.clamp(0.001,0.5)
        # 对logits进行温度缩放
        scaled_logits = logits / temperature
        # 确保正样本数至少为1，避免除零错误
        safe_num_pos = num_pos.clamp(min=1.0) 
        # 将标签转换为与缩放后logits相同的浮点类型
        aux_label_float = label.to(dtype=scaled_logits.dtype)
        # 计算InfoNCE损失：负的对数softmax概率与标签的加权和
        one_aux_loss = -torch.sum(F.log_softmax(scaled_logits, dim=0) * (aux_label_float / safe_num_pos), dim=0)
        return one_aux_loss

    def caculate_bce_loss(self, logits, label):
        label[label==0] = -1 # 将 0 (负样本) 映射为 -1
        one_aux_loss = -torch.mean(F.logsigmoid((logits + self.b) * label))
        return one_aux_loss

    def calculate_multi_pos_focal_infonce(self, logits, label, gamma=2.0):
        """
        Args:
            logits: (Batch_Size, Num_Candidates) 或者是 (N, 1) 的形式
            label: (Batch_Size, Num_Candidates) Multi-hot 标签，1为正，0为负
            gamma: Focal 参数
        """
        with torch.no_grad():
            temperature = self.temperature.clamp(0.001, 0.5)
        
        # 1. 缩放 logits
        scaled_logits = logits / temperature
        
        # 2. 核心 Trick：数值稳定地将 logits 转为 exp 域
        # 为了防止 exp 溢出，先减去最大值
        max_logits = torch.max(scaled_logits, dim=0, keepdim=True)[0].detach()
        exp_logits = torch.exp(scaled_logits - max_logits)
        
        # 3. 构建分母 (Denominator)
        # 传统的 InfoNCE 分母是 sum(exp_logits)
        # 我们现在的分母应该是：当前正样本的 exp + 所有负样本的 exp
        # 等价于：总和 - 其他正样本的 exp
        
        sum_exp = torch.sum(exp_logits, dim=0, keepdim=True) # 所有样本的 exp 之和
        
        # 利用 label (multi-hot) 找出所有正样本的 exp
        # label 为 1 的位置是正样本
        pos_exp = exp_logits * label 
        neg_exp_sum = torch.sum(exp_logits * (1 - label), dim=0, keepdim=True) # 所有负样本的 exp 之和
        
        # 4. 计算每个正样本对应的 Softmax 概率
        # 对于每一个位置 i (如果是正样本)：
        # Prob_i = exp_i / (exp_i + sum(exp_negatives))
        # 注意：这里分母不包含 label 中其他的正样本！
        
        # 这里利用广播机制：
        # 分母 = 当前位置的 exp (如果是正样本) + 所有负样本的 sum_exp
        denominators = exp_logits + neg_exp_sum
        
        # 计算概率 P
        probs = exp_logits / denominators
        
        # 5. 计算 Log Softmax (数值更稳定)
        # log(p) = logits - log(denominators)
        # 同样使用数值稳定的 logaddexp 或者直接操作
        # 这里为了代码清晰，直接对上面算出的 probs 取 log，实际工程中建议用 log_softmax 形式优化
        log_probs = torch.log(probs + 1e-10)
        
        # 6. Focal Weight
        # w = (1 - p)^gamma
        focal_weights = (1 - probs).pow(gamma)
        
        # 7. 计算 Loss
        # 只保留正样本位置的 Loss
        loss_map = - focal_weights * log_probs * label
        
        # 8. 归一化
        # 除以正样本的总数
        num_pos = label.sum()
        loss = loss_map.sum() / (num_pos + 1e-6)
        return loss

    def calculate_focal_infonce_loss(self, logits, label, num_pos, gamma=2.0):
        """
        Args:
            logits: 模型输出的 logits
            label: 正样本的 mask (通常是 multi-hot)
            num_pos: 正样本的数量
            gamma: Focal Loss 的超参数，控制挖掘难样本的程度，通常设为 2.0
        """
        with torch.no_grad():
            # 限制温度范围，防止数值不稳定
            temperature = self.temperature.clamp(0.001, 0.5)
        
        # 1. 缩放 Logits
        scaled_logits = logits / temperature
        
        # 2. 提前计算 Softmax 分数 (即公式中的 P = exp(s)/Z)
        # dim=0 对应你的输入是单个向量的情况
        probs = F.softmax(scaled_logits, dim=0)
        
        # 3. 计算 Log Softmax (用于 Loss 计算，比 log(probs) 数值更稳定)
        log_probs = F.log_softmax(scaled_logits, dim=0)
        
        # 4. 计算 Focal Weight: w = (1 - p)^gamma
        # 公式对应：w_i = (1 - p_i)^gamma
        # 这里不对权重进行 detach，允许梯度回传以调整对难易样本的关注度 (参考原始 Focal Loss 论文)
        focal_weights = (1 - probs).pow(gamma)
        
        # 5. 准备 Label 和归一化项
        safe_num_pos = num_pos.clamp(min=1.0) 
        aux_label_float = label.to(dtype=scaled_logits.dtype)
        
        # 6. 计算加权的 InfoNCE Loss
        # 公式对应：Sum [ -w_i * log(p_i) ]
        # 这里利用 label (0/1) 来只保留正样本的 Loss，并除以 num_pos 做平均
        weighted_loss = -torch.sum(
            focal_weights * log_probs * (aux_label_float / safe_num_pos), 
            dim=0
        )

        return weighted_loss

    def caculate_aux_loss(self, aux_loss, outputs, batch_aux_labels, device, dtype):
        count = 0
        all_layer_doc_score = outputs.all_docs_scores
        train_router_metrics = {}
        for layer_idx, layer_doc_score in enumerate(all_layer_doc_score):
            # if self.mid_layers <= layer_idx < self.num_layers:
            if self.mid_layers <= layer_idx and layer_idx in self.router_layer_idx:
                for b in range(len(batch_aux_labels)):
                    aux_logits_full = layer_doc_score[b] 
                    aux_label = torch.LongTensor(batch_aux_labels[b]).type(dtype).to(device)

                    # 1. 创建一个 mask 来找到所有有效的 (非 -inf) doc
                    valid_doc_mask = (aux_logits_full > -1e9) # 或 != -float('inf')

                    # 2. 使用 *同一个 mask* 来过滤 logits 和 labels
                    aux_logits = aux_logits_full[valid_doc_mask]
                    
                    if aux_label.shape[0] == 0:
                        continue
                    
                    num_pos = aux_label.sum()
                    if self.aux_loss_method == "BCE":
                        one_aux_loss = self.caculate_bce_loss(aux_logits, aux_label)
                    elif self.aux_loss_method == "INFONCE":
                        one_aux_loss = self.caculate_infonce_loss(aux_logits, aux_label, num_pos)
                    elif self.aux_loss_method == "INFONCE_FOCAL":
                        one_aux_loss = self.calculate_focal_infonce_loss(aux_logits, aux_label, num_pos)
                    elif self.aux_loss_method == "INFONCE_DECOUPLE":
                        one_aux_loss = self.calculate_decoupled_infonce_loss(aux_logits, aux_label, num_pos)
                    elif self.aux_loss_method == "INFONCE_DECOUPLE_FOCAL":
                        one_aux_loss = self.calculate_multi_pos_focal_infonce(aux_logits, aux_label, num_pos)

                    aux_loss += one_aux_loss
                    count += 1

                    # 计算recall
                    recall_at_n = [1, 5, 10]
                    for at_n in recall_at_n:
                        at_n = min(at_n, aux_logits.shape[0])
                        top_k_indices = torch.topk(aux_logits, at_n)[1].cpu().tolist()
                        hit_count = 0
                        for idx in top_k_indices:
                            if aux_label[idx] == 1:
                                hit_count += 1
                        recall_n = hit_count / min(num_pos.item(), 10)
                        if f'recall@{at_n}' not in train_router_metrics:
                            train_router_metrics[f'recall@{at_n}'] = [recall_n]
                        else:
                            train_router_metrics[f'recall@{at_n}'].append(recall_n)
        
        if count != 0:
            aux_loss = aux_loss / count
        for k, v in train_router_metrics.items():
            train_router_metrics[k] = sum(v) / len(v)
        return aux_loss, train_router_metrics