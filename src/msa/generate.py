import logging
import os
import re
from typing import Optional, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerateNonBeamOutput,
    GenerationMixin,
)

logger = logging.getLogger(__name__)

class MSAGenerationMixin(GenerationMixin):
    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        model_forward = self.__call__
        if isinstance(model_kwargs.get("past_key_values"), Cache):
            is_compileable = model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
            if getattr(self, "hf_quantizer", None) is not None:
                is_compileable &= self.hf_quantizer.is_compileable
            is_compileable = is_compileable and not generation_config.disable_compile
            if is_compileable and (
                self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
            ):
                os.environ["TOKENIZERS_PARALLELISM"] = "0"
                model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        meta = model_kwargs["past_key_values"].meta
        max_generate_tokens = model_kwargs["past_key_values"].meta["max_generate_tokens"]
        tokenizer = meta['tokenizer']
        response_string = meta["response_string"]
        idx_to_doc = meta["idx_to_doc"]
        pattern = meta["pattern"]
        retrieval_end_flags = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        round_end_flags = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        inner_string = ["" for _ in range(batch_size)]
        source_context_copied = False
        all_input_str = [""] * batch_size
        last_valid_inputs = input_ids[:, -1:].clone().to(input_ids.device)
        is_first = 1
        generate_stage = 1
        cnt = 0
        all_model_inputs = {}
        has_generate_stage3 = False
        first_stage2 = True
        round_end = False

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if "position_ids" in model_kwargs:
                position_ids = model_kwargs.pop("position_ids")
            else:
                position_ids = (position_ids[:, -1:] + model_kwargs["attention_mask"]).to(input_ids.device)

            model_inputs = model_kwargs.copy()

            if is_first == 1:
                model_inputs.update({"input_ids": input_ids})
            else:
                model_inputs.update({"input_ids": last_valid_inputs})

            model_inputs.update({"position_ids": position_ids})
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            input_str = tokenizer.batch_decode(model_inputs['input_ids'])
            all_input_str = [s + input_str[i] for i, s in enumerate(all_input_str)]

            if not is_prefill:
                for layer_idx in range(self.config.num_hidden_layers):
                    model_inputs["past_key_values"].record_kwargs(layer_idx, {"stage": "generate"})

            if 'doc_ids' not in all_model_inputs:
                all_model_inputs['doc_ids'] = model_inputs['doc_ids'].clone().to(input_ids.device)
            else:
                all_model_inputs['doc_ids'] = torch.cat((all_model_inputs['doc_ids'], model_inputs['doc_ids'][:, -model_inputs['attention_mask'].shape[1]:]), dim=1)

            all_model_inputs['attention_mask'] = torch.cat((all_model_inputs['attention_mask'], model_inputs['attention_mask']), dim=1) if 'attention_mask' in all_model_inputs else model_inputs['attention_mask'].clone().to(input_ids.device)
            all_model_inputs['input_ids'] = torch.cat((all_model_inputs['input_ids'], model_inputs['input_ids']), dim=1) if 'input_ids' in all_model_inputs else model_inputs['input_ids'].clone().to(input_ids.device)
            all_model_inputs['position_ids'] = torch.cat((all_model_inputs['position_ids'], model_inputs['position_ids']), dim=1) if 'position_ids' in all_model_inputs else model_inputs['position_ids'].clone().to(input_ids.device)
            all_model_inputs['past_key_values'] = model_inputs.get('past_key_values')
            all_model_inputs['cache_position'] = model_inputs['cache_position'].clone().to(input_ids.device)

            if round_end:
                break

            if is_prefill:
                outputs = self(**all_model_inputs, return_dict=True)
                is_prefill = False
                first_stage2 = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)
            is_first = 0

            # update model kwargs for next generation step
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            next_token_scores = logits_processor(last_valid_inputs, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            cur_generate_context = tokenizer.batch_decode(next_tokens)

            if generate_stage == 2 or generate_stage == 3:
                source_context_list = []
                imstart_str = "<|im_start|>"

                if generate_stage == 2:
                    for i, response_sample in enumerate(inner_string):
                        if '<End-of-Retrieve>' in response_sample:
                            question = all_input_str[i].split('historical document information\n\n')[-1]
                            question = question.split('\nPlease return all documents related to the question')[0]
                            source_context_list.append(imstart_str + 'The user\'s question is: %s\n<|object_ref_end|>' % (question))
                        else:
                            result = re.findall(pattern, response_sample)
                            indices = sorted(list(set(map(int, result))))
                            indices = [idx for idx in indices if idx in idx_to_doc]
                            try:
                                response_doc_str = ''.join(f"[{idx}]. {idx_to_doc[idx]}\n" for idx in indices)
                                response_doc_str = response_doc_str + '<|object_ref_end|>'
                            except KeyError as e:
                                logger.warning("Document not found for index %s, available indices: %s", e, indices)
                                response_doc_str = "" + '<|object_ref_end|>'
                            source_context_list.append(response_doc_str)

                if generate_stage == 3:
                    for i, response_sample in enumerate(response_string):
                        if '<End-of-Retrieve>' in response_sample:
                            question = all_input_str[i].split('historical document information\n\n')[-1]
                            question = question.split('\nPlease return all documents related to the question')[0]
                            source_context_list.append(imstart_str + 'The user\'s question is: %s\n<|object_ref_end|>' % (question))
                        else:
                            source_context_list.append("")

                source_batch = tokenizer(
                    source_context_list,
                    padding="longest",
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=True,
                    padding_side="left",
                )

                sh = source_batch['input_ids'].shape

                batch_source_input_ids = source_batch['input_ids'].clone().detach().long().to(input_ids.device)
                batch_source_attn_mask = source_batch['attention_mask'].clone().detach().long().to(input_ids.device)
                batch_source_doc_ids = torch.zeros_like(source_batch['input_ids'], dtype=torch.long, device=input_ids.device)
                batch_source_position_ids = torch.arange(sh[1], dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(sh[0], -1)
                batch_source_position_ids = batch_source_position_ids + model_inputs['position_ids'] + torch.sum(batch_source_attn_mask, dim=1, keepdim=True) - sh[1] + 1

                input_ids = batch_source_input_ids
                model_kwargs['attention_mask'] = batch_source_attn_mask
                model_kwargs['doc_ids'] = batch_source_doc_ids
                model_kwargs['position_ids'] = batch_source_position_ids
                source_context_copied = True

                cur_len += sh[1]
                this_peer_finished = False
                del outputs
                is_first = 1
                inner_string = ["" for _ in range(batch_size)]

                if generate_stage == 2:
                    round_end = True

                if generate_stage == 2:
                    generate_stage = 1

                if generate_stage == 3:
                    generate_stage = 4

                round_end_flags = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

            else:
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                if True:
                    temp = stopping_criteria(input_ids[:, -1:], scores)
                    unfinished_sequences = unfinished_sequences & ~temp
                    # max_res_len = 3000
                    # if max([len(n) for n in response_string]) > max_res_len:
                    #     unfinished_sequences = torch.zeros_like(unfinished_sequences)

                this_peer_finished = unfinished_sequences.max() == 0
                if this_peer_finished and not source_context_copied:
                    logger.warning("Generation finished before source context was copied")

                if streamer is not None:
                    streamer.put(next_tokens.cpu())
                cur_len += 1
                del outputs
                input_ids = input_ids[:, -1:]
                model_kwargs['doc_ids'] = torch.nn.functional.pad(model_kwargs['doc_ids'], (0, 1), value=-1)

                model_kwargs["attention_mask"] = torch.ones(batch_size, 1, dtype=torch.long, device=input_ids.device)
                if generate_stage == 1:
                    for i, retrieval_end_flag in enumerate(retrieval_end_flags):
                        if retrieval_end_flag or round_end_flags[i]:
                            model_kwargs["attention_mask"][i, 0] = 0
                        else:
                            last_valid_inputs[i, -1] = input_ids[i, -1]
                else:
                    for i, retrieval_end_flag in enumerate(retrieval_end_flags):
                        last_valid_inputs[i, -1] = input_ids[i, -1]

                assert len(cur_generate_context) == len(response_string)
                for i in range(len(cur_generate_context)):
                    response_string[i] += cur_generate_context[i]
                    inner_string[i] += cur_generate_context[i]
                    round_end_flags[i] |= "<|object_ref_end|>" in inner_string[i]
                    retrieval_end_flags[i] |= '<End-of-Retrieve>' in response_string[i]
                    if source_context_copied and generate_stage == 1:
                        mypattern = r"^\[\d*\]?$"
                        is_id = bool(re.fullmatch(mypattern, inner_string[i]))
                        is_EOR = inner_string[i] == '<End-of-Retrieve>'[:len(inner_string[i])]
                        round_end_flags[i] |= (not is_id and not is_EOR)

                    if not source_context_copied:
                        max_ret_len = 1000
                        retrieval_end_flags[i] |= len(response_string[i]) > max_ret_len

                if sum(round_end_flags * unfinished_sequences * (~retrieval_end_flags)) == sum(unfinished_sequences * (~retrieval_end_flags)) and not has_generate_stage3:
                    generate_stage = 2
                    for i in range(len(cur_generate_context)):
                        if source_context_copied and round_end_flags[i]:
                            if "<|object_ref_end|>" not in inner_string[i] or not bool(re.fullmatch(mypattern, inner_string[i].split("<|object_ref_end|>")[0])):
                                retrieval_end_flags[i] = True
                if sum(retrieval_end_flags * unfinished_sequences) == sum(unfinished_sequences) and not has_generate_stage3:
                    generate_stage = 3
                    has_generate_stage3 = True
            cnt += 1
            if cnt > max_generate_tokens:
                break

        all_model_inputs['attention_mask'], indices = all_model_inputs['attention_mask'].sort(dim=1, descending=False, stable=True)
        all_model_inputs['input_ids'] = all_model_inputs['input_ids'].gather(dim=1, index=indices)
        all_model_inputs['position_ids'] = all_model_inputs['position_ids'].gather(dim=1, index=indices)
        all_model_inputs['doc_ids'] = all_model_inputs['doc_ids'].gather(dim=1, index=indices)

        input_ids = all_model_inputs['input_ids']

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
