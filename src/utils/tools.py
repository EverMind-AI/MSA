import time
import torch
import threading

def format_bytes(size_in_bytes):
    """
    将字节数转换为人类可读的格式
    单位为 M 及以下不要小数点，单位为 G 以上保留一位小数，数值不能小于 1
    """
    # 定义单位
    units = ['B', 'K', 'M', 'G', 'T', 'P']
    
    # 处理边界情况
    if size_in_bytes < 1:
        return "0B"
    
    # 计算单位索引
    unit_index = 0
    size = float(size_in_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    # 根据单位决定格式化方式
    if unit_index <= 2:  # B, K, M 不要小数点
        if size == int(size):
            return f"{int(size)}{units[unit_index]}"
        else:
            return f"{int(round(size))}{units[unit_index]}"
    else:  # G 及以上保留一位小数
        return f"{size:.1f}{units[unit_index]}"

def cumulative_concat(tensors):
    # 一次性获取所有信息
    lengths = [len(t) for t in tensors]
    last_values = [t[-1] for t in tensors]
    
    # 计算累积偏移（不包括最后一个tensor）
    cum_offsets = torch.cumsum(torch.tensor([0] + last_values[:-1]), dim=0)
    
    # 构建偏移数组
    total_length = sum(lengths)
    offsets = torch.zeros(total_length, dtype=tensors[0].dtype, device=tensors[0].device)
    
    # 为每个tensor设置对应的偏移
    start_idx = 0
    for i, length in enumerate(lengths):
        if i > 0:  # 第一个tensor不需要偏移
            offsets[start_idx:start_idx + length] = cum_offsets[i]
        start_idx += length
    
    # 一次性拼接和添加偏移
    concatenated = torch.cat(tensors)
    return concatenated + offsets

    
class RequestLimiter:
    def __init__(self, max_concurrent=10):
        """
        初始化请求限流器
        
        Args:
            max_concurrent: 最大并发请求数，默认10
        """
        self.max_concurrent = max_concurrent
        self.current_count = 0
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
    
    def acquire(self):
        """
        获取执行权限，如果超过最大并发数则阻塞
        
        Returns:
            bool: 是否成功获取权限
        """
        with self.lock:
            while self.current_count >= self.max_concurrent:
                # 等待有请求完成
                self.condition.wait()
            self.current_count += 1
            return True
    
    def release(self):
        """
        释放一个执行权限，唤醒等待的请求
        """
        with self.lock:
            if self.current_count > 0:
                self.current_count -= 1
                # 通知一个等待的线程
                self.condition.notify()

def compose_input(doc, doc_idx, tokenizer):
    """组建reference的 input"""
    new_doc = "<|im_start|>" + f"[{doc_idx}]. {doc}[{doc_idx}]<|im_end|>"
    return new_doc,tokenizer(new_doc, add_special_tokens=False)


class TimePoint:
    def __init__(self, disabled=False):
        self.pts = []
        self.disabled = disabled
    
    def add(self, name):
        if not self.disabled:
            self.pts.append((name, time.time()))

    def print(self):
        if len(self.pts) < 2:
            return

        total = self.pts[-1][1] - self.pts[0][1]
        s = ""
        if len(self.pts) > 2:
            lst1 = self.pts[:-1]
            lst2 = self.pts[1:]
            s = " | ".join(f"{item1[0]}->{item2[0]}: {item2[1] - item1[1]:.3f}" for item1, item2 in zip(lst1, lst2))
        print(f"total: {total:.2f} {s}")