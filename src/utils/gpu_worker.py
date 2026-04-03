import os
import torch

class GpuWorker:

    def __init__(self, gpu_id: int, envs: dict):
        self.gpu_id = gpu_id
        self._setup_environment(envs)
        self.device = self._setup_device()

    def _setup_environment(self, envs):
        """设置环境变量"""
        for k, v in envs.items():
            os.environ[k] = v

    def _setup_device(self):
        """设置CUDA设备"""
        torch.cuda.set_device(self.gpu_id)
        return f"cuda:{self.gpu_id}"