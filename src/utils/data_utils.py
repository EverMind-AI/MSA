import os
import lmdb
from multiprocessing import Pool
from tqdm import tqdm
import importlib.metadata
import importlib.util
from packaging import version
from typing import TYPE_CHECKING
from functools import lru_cache
from datasets import Dataset, IterableDataset
from src.utils.common import IGNORE_INDEX
from collections import defaultdict
from functools import partial
import bisect
from typing import List, Sequence, Tuple, Optional, Union
from src.utils.common import pdb_debug

if TYPE_CHECKING:
    from packaging.version import Version

def write_lmdb(output_dir, name):

    os.makedirs(output_dir, exist_ok=True)
    output_name = os.path.join(output_dir, f'{name}.lmdb')

    try:
        os.remove(output_name)
    except:
        pass
    env_new = lmdb.open(
        output_name,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn_write = env_new.begin(write=True)

    return txn_write, env_new

def read_lmdb(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    return env, txn


def get_length(index):
    return index, len(global_dataset[index]["input_ids"])

def get_sequence_length(dataset, num_worker=16):
    global global_dataset
    global_dataset = dataset
    num_data = len(dataset)
    lengths = [0] * num_data
    with Pool(processes=num_worker) as pool:
        iters = pool.imap(get_length, range(num_data))
        for i, length in tqdm(iters, total=num_data):
            lengths[i] = length
    return lengths

def get_data(index):
    item = global_torch_dataset[index]
    length = len(global_torch_dataset[index]["input_ids"])
    return item, index, length

def torch_dataset_to_hf_dataset(torch_dataset, num_worker=16):
    global global_torch_dataset
    global_torch_dataset = torch_dataset
    num_data = len(global_torch_dataset)
    lengths = [0] * num_data
    hf_dict = {key: [] for key in torch_dataset[0].keys()}
    with Pool(processes=num_worker) as pool:
        iters = pool.imap(get_data, range(num_data))
        for data, i, length in tqdm(iters, total=num_data):
            for key, value in data.items():
                hf_dict[key].append(value)
            lengths[i] = length
    hf_dataset = Dataset.from_dict(hf_dict)
    return hf_dataset, lengths

def _get_package_version(name: str) -> "Version":
    try:
        return version.parse(importlib.metadata.version(name))
    except Exception:
        return version.parse("0.0.0")

@lru_cache
def is_transformers_version_greater_than(content: str):
    return _get_package_version("transformers") >= version.parse(content)

@lru_cache
def is_transformers_version_equal_to_4_46():
    return version.parse("4.46.0") <= _get_package_version("transformers") <= version.parse("4.46.1")

def search_for_fit(numbers: Sequence[int], capacity: int) -> int:
    r"""
    Finds the index of largest number that fits into the knapsack with the given capacity.
    """
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)

def greedy_knapsack(numbers: List[int], capacity: int) -> List[List[int]]:
    r"""
    An efficient greedy algorithm with binary search for the knapsack problem.
    """
    numbers.sort()  # sort numbers in ascending order for binary search
    knapsacks = []

    while numbers:
        current_knapsack = []
        remaining_capacity = capacity

        while True:
            index = search_for_fit(numbers, remaining_capacity)
            if index == -1:
                break  # no more numbers fit in this knapsack

            remaining_capacity -= numbers[index]  # update the remaining capacity
            current_knapsack.append(numbers.pop(index))  # add the number to knapsack

        knapsacks.append(current_knapsack)

    return knapsacks

def preprocess_packed_supervised_dataset(examples, tokenizer, cutoff_len):
    valid_num = 0
    batch_input_ids, batch_labels = [], []
    lengths = []
    length2indexes = defaultdict(list)
    for i in range(len(examples["input_ids"])):
        input_ids, labels = examples["input_ids"][i], examples["labels"][i]
        length = len(input_ids)
        if length >= cutoff_len - 1:
            continue
        else:
            lengths.append(length)
            length2indexes[length].append(valid_num)
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            valid_num += 1
    model_inputs = defaultdict(list)
    knapsacks = greedy_knapsack(lengths, cutoff_len - 1)
    for knapsack in knapsacks:
        packed_input_ids, packed_attention_masks, packed_labels = [], [], []
        for i, length in enumerate(knapsack):
            index = length2indexes[length].pop()
            packed_input_ids += batch_input_ids[index]
            packed_labels += batch_labels[index]
            packed_attention_masks += [1] * len(batch_input_ids[index])

        if len(packed_input_ids) < cutoff_len:
            pad_length = cutoff_len - len(packed_input_ids)
            packed_input_ids += [tokenizer.pad_token_id] * pad_length
            packed_labels += [IGNORE_INDEX] * pad_length
            packed_attention_masks += [1] * pad_length  # more efficient flash_attn

        if len(packed_input_ids) != cutoff_len:
            raise ValueError("The length of packed example should be identical to the cutoff length.")

        model_inputs["input_ids"].append(packed_input_ids)
        model_inputs["attention_mask"].append(packed_attention_masks)
        model_inputs["position_ids"].append(list(range(len(packed_input_ids))))
        model_inputs["labels"].append(packed_labels)
    return model_inputs

def pad_sequence(examples, cutoff_len, tokenizer):
    max_length = cutoff_len
    input_pad_token_id = tokenizer.pad_token_id
    label_pad_token_id = IGNORE_INDEX

    for k, v in examples.items():
        if k.endswith("input_ids"):
            pad_token_id = input_pad_token_id
        elif k.endswith("labels"):
            pad_token_id = label_pad_token_id
            # shift labels here
            for i in range(len(v)):
                v[i] = v[i][1:]
        elif k.endswith("attention_mask"):
            pad_token_id = 0
        elif k.endswith("position_ids"):
            pad_token_id = max_length - 1  # pad the max position id
        elif k == "images" or k == "videos":
            pad_token_id = -1
            continue  # TODO: haven't tested multi-modal yet
        else:
            continue
        for i in range(len(v)):
            v[i].extend([pad_token_id] * (max_length - len(v[i])))
        examples[k] = v

    return examples

def preprocess_sp_dataset(seq_ids, world_size, sequence_parallel_mode):
    if sequence_parallel_mode == "zigzag-ring":
        step = len(seq_ids) // (2 * world_size)
        value_chunks = [seq_ids[s : s + step] for s in range(0, len(seq_ids), step)]
        local_values = list()
        for rank in range(world_size):
            local_values.append(value_chunks[rank] + value_chunks[2 * world_size - rank - 1])
        return local_values
    elif sequence_parallel_mode == "ulysses":
        step = len(seq_ids) // world_size
        local_values = [seq_ids[s : s + step] for s in range(0, len(seq_ids), step)]
        return local_values
    else:
        raise NotImplementedError("Other sequence parallel modes are to be implemented.")


# sp for Sequence Parallel
def sp_split(examples, sequence_parallel_size, sequence_parallel_mode="ulysses"):
    for k, v in examples.items():
        chunks = list()
        for row in v:
            if k.endswith("attention_mask"):
                chunks.extend([row] * sequence_parallel_size)
            elif row is None:
                chunks.extend([None] * sequence_parallel_size)
            else:
                chunks.extend(
                    preprocess_sp_dataset(row, sequence_parallel_size, sequence_parallel_mode)
                )
        examples[k] = chunks
    return examples

def get_sequence_parallel_preprocess(stage, tokenizer, cutoff_len=None, sequence_parallel_size=1, sequence_parallel_mode="ulysses"):
    if stage == "pad":
        assert cutoff_len is not None
        preprocess_func = partial(pad_sequence, cutoff_len=cutoff_len, tokenizer=tokenizer)
    elif stage == "split":
        preprocess_func = partial(sp_split, sequence_parallel_size=sequence_parallel_size, sequence_parallel_mode=sequence_parallel_mode)
    else:
        raise NotImplementedError(f"Unexpected stage in sequence_parallel_preprocess: {stage}")

    return preprocess_func

def _get_sequence_parallel_dataset(dataset, num_works, tokenizer=None, cutoff_len=10000, 
                                   sequence_parallel_size=1, sequence_parallel_mode="ulysses", 
                                   cache_dataset_overwrite=False) -> Optional[Union["Dataset", "IterableDataset"]]:
    kwargs = dict(
        num_proc=num_works,
        load_from_cache_file=not cache_dataset_overwrite,
        desc="Running padding split on dataset",
    )
    pad_sequence_func = get_sequence_parallel_preprocess(
        stage="pad", 
        tokenizer=tokenizer, 
        cutoff_len=cutoff_len
    )
    padded_dataset = dataset.map(
        pad_sequence_func, batched=True, batch_size=num_works, **kwargs
    )
    kwargs = dict(
        num_proc=num_works,
        load_from_cache_file=not cache_dataset_overwrite,
        desc="Running sequence parallel split on dataset",
    )
    sp_dataset_func = get_sequence_parallel_preprocess(
        stage="split", 
        tokenizer=tokenizer, 
        sequence_parallel_size=sequence_parallel_size, 
        sequence_parallel_mode=sequence_parallel_mode,
    )
    sp_dataset = padded_dataset.map(
        sp_dataset_func, batched=True, batch_size=num_works, **kwargs
    )
    return sp_dataset

def packing_dataset(dataset, tokenizer, cutoff_len, num_worker, cache_dataset_overwrite):
    preprocess_func = partial(preprocess_packed_supervised_dataset, tokenizer=tokenizer, cutoff_len=cutoff_len)
    kwargs = dict(
        num_proc=num_worker,
        load_from_cache_file=not cache_dataset_overwrite,
        desc="Running postprocess on dataset",
    )
    import pdb; pdb.set_trace()
    dataset = dataset.map(
        preprocess_func,
        batched=True,
        batch_size=num_worker,
        **kwargs,
    )
    return dataset

def data_post_process_sequence_parallel(
    dataset, 
    training_args, 
    sequence_parallel_size, 
    sequence_parallel_mode, 
    cutoff_len, 
    num_worker=16, 
    packing=False, 
    tokenizer=None,
    cache_dataset_overwrite=False,
):
    dataset = dataset.shuffle(seed=training_args.seed)
    if packing:
        dataset = packing_dataset(dataset, tokenizer, cutoff_len, num_worker, cache_dataset_overwrite)

    dataset = _get_sequence_parallel_dataset(dataset, 
                                             num_works=num_worker, 
                                             tokenizer=tokenizer, 
                                             cutoff_len=cutoff_len, 
                                             sequence_parallel_size=sequence_parallel_size,
                                             sequence_parallel_mode=sequence_parallel_mode,
                                             cache_dataset_overwrite=cache_dataset_overwrite)
    return dataset