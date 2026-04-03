import os
import time
import typing
# from deepspeed.accelerator import get_accelerator

# NOTE: 最新版开始迁移到 integration_utils
try:
    from transformers.integrations import TrainerCallback
except ImportError:
    from transformers.integrations.integration_utils import TrainerCallback

# USE_FLASH_ATTN, USE_XFORMERS_ATTN = False, False
# if os.getenv('FLASH_ATTN', 'false').lower() == 'true':
#     USE_FLASH_ATTN = True
#     from mllm.utils.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn, restore_llama_attn
# if os.getenv("XFORMERS_ATTN", 'false').lower() == 'true':
#     USE_XFORMERS_ATTN = True
#     from mllm.utils.llama_xformers_monkey_patch import replace_llama_attn_with_xformers_attn, restore_llama_attn

class ModeltimeCallback(TrainerCallback):
    def __init__(self):
        self.model_time = 0.
        self.data_time = 0.
        self._start = 0.
        self._end = 0.

    def on_train_begin(self, args, state, control, **kwargs):
        self._end = time.time()
    
    def on_step_begin(self, args, state, control, **kwargs):
        self._start = time.time()
        self.data_time += (self._start - self._end)

    def on_step_end(self, args, state, control, **kwargs):
        self._end = time.time()
        self.model_time += (self._end - self._start)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        
        data_time = self.data_time / state.logging_steps
        model_time = self.model_time / state.logging_steps
        self.data_time = 0.
        self.model_time = 0.
        
        info = f'\nSTEP: {state.global_step}, data_time: {data_time:.3f}, model_time: {model_time:.3f}'
        print(info)


class SacredCallback(TrainerCallback):
    def __init__(self, _run=None):
        self._run = _run
        self.model_time = 0.
        self.data_time = 0.
        self._start = 0.
        self._end = 0.
        self._zero_loss_cnt = 0
    
    def on_train_begin(self, args, state, control, **kwargs):
        if self._run:
            self._end = time.time()
    
    def on_step_begin(self, args, state, control, **kwargs):
        if self._run:
            self._start = time.time()
            self.data_time += (self._start - self._end)

    def on_step_end(self, args, state, control, **kwargs):
        if self._run:
            self._end = time.time()
            self.model_time += (self._end - self._start)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        
        if self._run is None:
            return 

        data_time = self.data_time / state.logging_steps
        model_time = self.model_time / state.logging_steps
        self.data_time = 0.
        self.model_time = 0.
        
        self._run.log_scalar("loss", logs.get("loss", 0.), state.global_step)
        self._run.log_scalar("learning_rate", logs.get("learning_rate", 0.), state.global_step)
        self._run.log_scalar("grad_norm", logs.get("grad_norm", 0.), state.global_step)
        self._run.log_scalar("epoch", state.epoch, state.global_step)
        self._run.log_scalar("data_time", data_time, state.global_step)
        self._run.log_scalar("model_time", model_time, state.global_step)
        
        if logs.get("loss", 0.) == 0.:
            self._zero_loss_cnt += 1
            if self._zero_loss_cnt > 1:
                raise RuntimeError("Loss is zero, something is wrong!")
        

class ModelEvalCallback(TrainerCallback):
    def __init__(self, _run=None, multitest=None, trainer=None, gen_kwargs=None):
        self._run = _run
        # datasets dict
        # key1: dataset_name
        # val1: {'dataset': dataset inst, 'compute_metric': metric inst}
        self.multitest = typing.cast(dict, multitest)
        self.trainer = trainer
        self.gen_kwargs = gen_kwargs
    
    def on_step_end(self, args, state, control, **kwargs):
        if args.eval_steps is None:
            eval_steps = args.save_steps
        elif isinstance(args.eval_steps, int) and args.eval_steps > 0:
            eval_steps = args.eval_steps
        else:
            return
        if state.global_step > 0 and state.global_step % eval_steps == 0:
            if not args.do_multi_predict:
                return
            
            # flash-attn currently not supports eval mode!
            # if USE_FLASH_ATTN or USE_XFORMERS_ATTN:
            #     restore_llama_attn()
            
            old_compute_metrics = self.trainer.compute_metrics
            
            for dataset_idx, (dataset_name, item) in enumerate(self.multitest.items()):
                print(f'processing multitest set {dataset_idx}/{len(self.multitest)}: {dataset_name}')
                _ds = item['dataset']
                _compute_metrics = item['compute_metric']
                _prefix = dataset_name
                
                self.trainer.compute_metrics = _compute_metrics
                # transformers.trainer_utils.PredictionOutput
                _pred_results = self.trainer.predict(_ds, metric_key_prefix=_prefix, **self.gen_kwargs)
                if state.is_world_process_zero:
                    self.trainer.log_metrics(_prefix, _pred_results.metrics)  # noqa
                    self.trainer.save_metrics(_prefix, _pred_results.metrics)  # noqa
                    self.trainer.save_prediction(_pred_results, file_key_prefix=_prefix)
                    
                    if self._run is not None:
                        keywords_to_remove = ['runtime', 'second']
                        for k, v in _pred_results.metrics.items():
                            # remove time releated metrics
                            if any(kw in k for kw in keywords_to_remove):
                                continue
                            self._run.log_scalar(f'{k}', v, state.global_step)
            
            self.trainer.compute_metrics = old_compute_metrics
            
            # if USE_FLASH_ATTN:
            #     replace_llama_attn_with_flash_attn()
            # if USE_XFORMERS_ATTN:
            #     replace_llama_attn_with_xformers_attn()

class DSEmptyCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        empty_cache_steps = int(os.getenv("EMPTY_CACHE_STEP", '0').strip())
        can_flush = state.global_step > 0 and empty_cache_steps > 0 and state.global_step % empty_cache_steps == 0
        if can_flush:
            # print('Flush Cache here.')
            get_accelerator().empty_cache()
            

# usage: https://github.com/yqhu/profiler-workshop/blob/c8d4a7c30a61cc7b909d89f88f5fd36b70c55769/hf_training_trainer_prof.py#L49C6-L49C28
# additionally, with_modules can be set True, with_flops must be set False
# deps: pip install -U tensorboard-plugin-profilepip torch_tb_profiler 
class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()
