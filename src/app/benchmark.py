import argparse
from collections import OrderedDict
import pickle
import re
from typing import Dict, List
from tqdm import tqdm
import json
import torch
import numpy as np
import pathlib
import sys
import os
import random
import numpy as np
import multiprocessing as mp

project_path = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(project_path))

from src.benchmarks import BenchMarks


def set_seed(seed):
    """
    固定所有随机种子以确保实验的可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 适用于多GPU环境
    os.environ['PYTHONHASHSEED'] = str(seed)

# 在程序的开始部分调用此函数
seed_value = 42
set_seed(seed_value)

def parse_benchmark_file(args):
    benchmark = BenchMarks(bench_name=args.benchmark)
    path, mem_path = benchmark.get_bench_files()
    args.query_file = path
    args.memory_file = mem_path
    print(" ==========> load query file from:", path)
    if path.endswith(".json"):
        with open(path, "r") as f:
            raw_data = json.load(f)
        ## 重组数据集
        data = [{
            'question': d['question'],
            'labels': d['labels'],
            'answer': d['answer']
        } for d in raw_data]
    # 判断是否是一个路径
    elif path.endswith("pkl"):
        with open(path, "rb") as f:
            query_metas = pickle.load(f)

        data = [{
            'question': q_meta['query'],
            'labels': q_meta['reference_list'],
            'answer': q_meta['answer']
        } for q_meta  in query_metas]
    else:
        raise ValueError(f"Unsupported file format: {path}")
    print(f"num sample: {len(data)}")
    return data

def sort_requests(data_items: List[Dict]) -> List[dict]:
    """
    创建动态批次：按照input_id长度排序，然后根据max_input_length分批
    """
    print("Tokenizing and sorting data for dynamic batching...")

    # 计算每个item的input_id长度
    items_with_length = []
    for item in tqdm(data_items):
        prompt = item["question"]
        length = len(prompt)

        items_with_length.append((item, length))

    # 按长度排序
    items_with_length.sort(key=lambda x: x[1])
    return [item[0] for item in items_with_length]


def base_it(predict, label, at, score_func):
    assert len(predict) == len(label)
    scores = []
    for pred, lbs in zip(predict, label):
        pred = pred.tolist() if not isinstance(pred, list) else pred
        best_score = 0.
        if not isinstance(lbs, list):
            lbs = [lbs]
        for lb in lbs:
            if isinstance(lb, list):
                lb = lb[0]
            rank = pred[:at].index(lb) + 1 if lb in pred[:at] else 0
            cur_score = score_func(rank)
            best_score = max(best_score, cur_score)
        scores.append(best_score)
    return scores


def eval_recall(predict, label, at=10):
    scores = base_it(predict, label, at, lambda rank: int(rank != 0))
    return {f'R@{at}': sum(scores) / len(scores)}


def eval_mrr(predict, label, at=10):
    scores = base_it(predict, label, at, lambda rank: 1 / rank if rank != 0 else 0)
    return {f'MRR@{at}': sum(scores) / len(scores)}


def eval_all(predict, label):
    log_dict = {}
    log_dict.update(eval_recall(predict, label, at=1))
    log_dict.update(eval_recall(predict, label, at=5))
    log_dict.update(eval_recall(predict, label, at=10))
    log_dict.update(eval_mrr(predict, label, at=1))
    return log_dict

def calculate_ir_metrics(true_labels: List[int], pred_labels: List[int]):
    """计算信息检索中的 Precision, Recall, F1, 和 IoU。"""
    true_set = set(true_labels)
    pred_set = set(pred_labels)
    if not true_set: return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'iou': 0.0}
    tp = len(true_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(true_set)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = len(true_set & pred_set) / len(true_set | pred_set) if len(true_set | pred_set) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou}

def process_results(requests: List[Dict], index_to_doc, doc_to_index):
    num_error = 0
    record_list = []
    all_metrics = {"precision": [], "recall": [], "f1": [], "iou": []}

    for request in requests:
        question = request["question"]
        answer = request["answer"]
        generated_text = request["response"].replace('<|endoftext|>', '')
        generated_text = "\nPlease answer the question based" + generated_text.split("\nPlease answer the question based")[1]

        labels = [doc_to_index[txt] for txt in request["labels"]]
        predictions = list(set(map(int, re.findall(r'\[(\d+)\]', generated_text))))

        try:
            pred_answer = generated_text.split('The answer to the question is:')[-1].split("<|im_end|>")[0].strip()
        except Exception as e:
            pred_answer = ""
            raise ValueError(f"输出格式异常: {e}, generated_text: {generated_text}")

        try:
            record_list.append({
                "labels_id": labels,
                "pred_id": predictions,
                "question": question,
                "true_answer": answer,
                "pred_answer": pred_answer,
                "generated_text": generated_text,
                "predict_context": [{i: index_to_doc[pid]} for i, pid in enumerate(predictions)],
                "gt_context": [{i: index_to_doc[pid]} for i, pid in enumerate(labels)],
            })
        except Exception as e:
            num_error += 1

        metrics = calculate_ir_metrics(labels, predictions)
        for k, v in metrics.items():
            all_metrics[k].append(v)

    metrics_dict = {k: round(float(np.mean(v)), 4) for k, v in all_metrics.items()}
    print(" ==================== Retrieve metrics ======================= ")
    print("AR Metrics: ", metrics_dict)
    print(" ==================== Retrieve metrics ======================= ")
    return {"metrics": metrics_dict, "record_list": record_list}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str)
    parser.add_argument('--max_batch_size', type=int)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--block_size', type=int, default=2048) # tokens for one memory inference
    parser.add_argument('--max_chunk_per_block', type=int, default=16*1024) # chunks per block slice
    parser.add_argument('--max_length', type=int, default=64) # max output length
    parser.add_argument('--max_seq_len', type=int, default=0) # max input+output length
    parser.add_argument('--max_query_seq_len', type=int, default=0) # max input seq len
    parser.add_argument('--template', type=str, default="QWEN3_TEMPLATE")
    parser.add_argument('--output_file', type=str, default="")
    parser.add_argument('--case_name', type=str, default="anonymous")

    args = parser.parse_args()

    return args


def should_regenerate(request:dict, response: str):
    """return a text if the response should be regenerated, or return None"""

    if response[-len("<|object_ref_end|>"):] == "<|object_ref_end|>" and "The answer to the question is:" not in response:
        if not request.get('regenerated', False):
            request["regenerated"] = True
        response = "\nPlease answer the question based"+response.split("\nPlease answer the question based")[1]
        response = response.replace('<|endoftext|>', '')
        return "<regenerate>"+ response

    return None

def read_config_to_args(config_path):
    with open(os.path.join(config_path, "config.json"), 'r') as f:
        msa_config = json.load(f).get("msa_config")
    args.doc_top_k = msa_config.get("doc_top_k", 16)
    args.pooling_kernel_size = msa_config.get("pooling_kernel_size", 64)
    args.router_layer_idx = msa_config.get("router_layer_idx", "all")
    return args, msa_config


def msa_benchmark(args, data):

    from src.msa_service import GenerateConfig, ModelConfig, MemoryConfig, MSAEngine
    args, msa_config = read_config_to_args(args.model_path)

    model_config = ModelConfig(model_path=args.model_path,
                                doc_top_k=args.doc_top_k,
                                pooling_kernel_size=args.pooling_kernel_size,
                                router_layer_idx=args.router_layer_idx,
                                )
    generate_config = GenerateConfig(devices=list(range(torch.cuda.device_count())),
                                     template=args.template,
                                     max_generate_tokens=args.max_length,
                                     max_seq_len=args.max_seq_len,
                                     max_query_seq_len=args.max_query_seq_len,
                                     max_batch_size=args.max_batch_size,
                                     top_p=args.top_p,
                                     temperature=args.temperature,
                                     qa_mode=True)
    memory_config = MemoryConfig(block_size=args.block_size,
                                 pooling_kernel_size=args.pooling_kernel_size,
                                 slice_chunk_size=args.max_chunk_per_block,
                                 memory_file_path=args.memory_file,
                                 )
    
    final_result = {}
    if args.output_file:
        try:
            with open(args.output_file, 'r') as f:
                exist_result = json.load(f)
                final_result = exist_result[args.case_name]
        except:
            pass

    with MSAEngine(generate_config, model_config, memory_config) as engine:
        print("start precision test")
        idx_to_doc = engine.get_idx_to_doc()
        doc_to_idx = {v: k for k, v in idx_to_doc.items()}

        bsz = args.max_batch_size * generate_config.world
        sorted_requests = sort_requests(data)
        requests = OrderedDict({idx: item for idx, item in enumerate(sorted_requests)})

        for req_idx in requests:
            requests[req_idx]['idx'] = req_idx

        total = len(requests)
        results = [] # processed requests
        pbar = tqdm(total=total, desc=f"Precision Test")

        to_send = {} 
        while len(results) < total:
            num = min(bsz-len(to_send), len(requests))
            for _ in range(num):
                idx, request = requests.popitem(last=False)
                to_send[idx] = request
            
            prompts, indices = [], [] 
            for idx, request in to_send.items():
                prompts.append(request.get("new_question", request["question"]))
                indices.append(idx)

            texts, recall_topks, _ = engine.generate(prompts, require_recall_topk=True)

            for idx, response in enumerate(texts):
                req_idx = indices[idx]
                request = to_send[req_idx]
                response = "\nPlease answer the question based"+response.split("\nPlease answer the question based")[1]
                response = response.replace('<|endoftext|>', '')
                new_prompt = should_regenerate(request, response)

                # 判断是否需要重新生成
                if new_prompt is not None:
                    request["new_question"] = new_prompt
                else:
                    recall_topk = {layer: v[idx] for layer, v in recall_topks.items()}
                    request = to_send.pop(req_idx)
                    request['recall_topk'] = recall_topk
                    request["response"] = response
                    results.append(request)
            pbar.update(bsz - len(to_send))
        pbar.close()

        assert len(results) == total, \
            f"Results count mismatch: got {len(results)}, expected {total} (from query_file)"

        final_result['precision'] = process_results(results, idx_to_doc, doc_to_idx)

    if args.output_file:
        exist_result = {}
        try:
            with open(args.output_file, 'r') as f:
                exist_result = json.load(f)
        except:
            pass
        exist_result[args.case_name] = final_result
        with open(args.output_file, 'w') as f:
            json.dump(exist_result, f, indent=4, ensure_ascii=False)
    else:
        s = json.dumps(final_result, indent=4, ensure_ascii=False)
        print(s)

if __name__ == "__main__":
    mp.set_start_method('spawn') 
    args = parse_args()
    assert args.template in ["QWEN3_TEMPLATE", "QWEN3_INSTRUCT_TEMPLATE"]
    if args.output_file:
        assert args.case_name != "", "when output result to a file, please give this test case a name"

    print(json.dumps(vars(args), indent=4, sort_keys=True))

    data = parse_benchmark_file(args)
    msa_benchmark(args, data)