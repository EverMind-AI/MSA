import pynvml
import time
import threading
from datetime import datetime
import sys
import os

class GPUMemoryMonitor:
    def __init__(self, gpu_index=0, interval=1.0, unit='GB'):
        """
        初始化GPU显存监控器
        
        Args:
            gpu_index: 要监控的GPU索引，默认0
            interval: 监控间隔时间（秒），默认1秒
            unit: 返回的单位，支持 'MB' 或 'GB'，默认'GB'
        """
        self.gpu_index = gpu_index
        self.interval = interval
        self.unit = unit.upper()
        
        # 验证单位参数
        if self.unit not in ['MB', 'GB']:
            raise ValueError("单位必须是 'MB' 或 'GB'")
        
        # 监控相关状态
        self.monitor_thread = None
        self._running = False
        self._lock = threading.Lock()
        self.peak_memory_usage = 0  # 峰值显存使用量
        self.start_time = None
        self.stop_time = None
        
        # 初始化NVML
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            if self.gpu_index >= self.device_count:
                raise ValueError(f"GPU索引 {self.gpu_index} 超出范围，系统只有 {self.device_count} 个GPU")
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            
            # 获取GPU名称
            self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
            
        except Exception as e:
            print(f"初始化NVML失败: {e}")
            if 'pynvml' in sys.modules:
                pynvml.nvmlShutdown()
            raise
    
    def _get_memory_info(self):
        """获取当前GPU显存信息"""
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return mem_info
        except Exception as e:
            print(f"获取GPU {self.gpu_index} 显存信息失败: {e}")
            return None
    
    def _convert_units(self, bytes_value):
        """转换字节为指定单位"""
        if self.unit == 'MB':
            return bytes_value / (1024 * 1024)
        else:  # GB
            return bytes_value / (1024 * 1024 * 1024)
    
    def _get_current_usage(self):
        """获取当前显存使用量（按指定单位）"""
        mem_info = self._get_memory_info()
        if mem_info:
            return self._convert_units(mem_info.used)
        return 0
    
    def _monitor_loop(self):
        """监控循环，运行在单独线程中"""
        # print(f"开始监控 GPU {self.gpu_index} ({self.gpu_name})")
        # print(f"监控间隔: {self.interval}秒")
        # print(f"单位: {self.unit}")
        # print("按回车键停止监控...\n")
        
        try:
            while self._running:
                # 获取当前显存使用量
                current_usage = self._get_current_usage()
                
                # 更新峰值
                with self._lock:
                    if current_usage > self.peak_memory_usage:
                        self.peak_memory_usage = current_usage
                
                # 等待下一个监控周期
                time.sleep(self.interval)
                
        except Exception as e:
            print(f"监控线程出错: {e}")
        finally:
            # print(f"GPU {self.gpu_index} 监控线程结束")
            pass
    
    def start(self):
        """
        启动显存监控
        
        Returns:
            bool: 是否成功启动
        """
        if self._running:
            # print(f"GPU {self.gpu_index} 监控已经在运行中")
            return False
        
        try:
            # 重置峰值数据
            with self._lock:
                self.peak_memory_usage = 0
            
            # 记录开始时间
            self.start_time = datetime.now()
            
            # 启动监控线程
            self._running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            # 验证线程已启动
            time.sleep(0.1)
            if self.monitor_thread.is_alive():
                # print(f"GPU {self.gpu_index} 监控已启动")
                return True
            else:
                self._running = False
                # print(f"GPU {self.gpu_index} 监控线程启动失败")
                return False
                
        except Exception as e:
            print(f"启动GPU {self.gpu_index} 监控失败: {e}")
            self._running = False
            return False
    
    def stop(self, verbose=False):
        """
        停止显存监控并返回峰值显存使用量
        
        Args:
            verbose: 是否打印详细信息
            
        Returns:
            float: 峰值显存使用量（单位：GB或MB）
        """
        if not self._running:
            if verbose:
                print(f"GPU {self.gpu_index} 监控未在运行")
            return 0.0
        
        try:
            # 停止监控线程
            self._running = False
            self.stop_time = datetime.now()
            
            # 等待线程结束（最多等待2秒）
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2.0)
            
            # 获取最终峰值
            peak_usage = 0.0
            with self._lock:
                peak_usage = self.peak_memory_usage
            
            if verbose:
                duration = (self.stop_time - self.start_time).total_seconds()
                print(f"\n{'='*50}")
                print(f"GPU {self.gpu_index} ({self.gpu_name}) 监控结果")
                print(f"{'='*50}")
                print(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"结束时间: {self.stop_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"监控时长: {duration:.1f} 秒")
                print(f"峰值显存使用量: {peak_usage:.3f} {self.unit}")
                
                # 获取当前显存信息作为对比
                mem_info = self._get_memory_info()
                if mem_info:
                    total = self._convert_units(mem_info.total)
                    current = self._convert_units(mem_info.used)
                    print(f"当前显存使用: {current:.3f} / {total:.3f} {self.unit}")
                    print(f"峰值占比: {(peak_usage/total*100):.1f}%")
                print(f"{'='*50}")
            
            return peak_usage
            
        except Exception as e:
            print(f"停止GPU {self.gpu_index} 监控时出错: {e}")
            return 0.0
    
    def get_current_usage(self):
        """获取当前显存使用量（不停止监控）"""
        return self._get_current_usage()
    
    def get_peak_usage(self):
        """获取当前记录的峰值显存使用量（不停止监控）"""
        with self._lock:
            return self.peak_memory_usage
    
    def is_running(self):
        """检查监控是否在运行"""
        return self._running
    
    def __del__(self):
        """析构函数，确保清理资源"""
        if self._running:
            self.stop(verbose=False)
        try:
            pynvml.nvmlShutdown()
        except:
            pass


# 使用示例函数
def monitor_gpu_memory_example():
    """使用示例"""
    print("GPU显存监控示例")
    print("=" * 50)
    
    try:
        # 创建监控器
        monitor = GPUMemoryMonitor(
            gpu_index=0,      # 监控第一个GPU
            interval=0.5,     # 每0.5秒检查一次
            unit='GB'         # 使用GB作为单位
        )
        
        # 启动监控
        if monitor.start():
            # 这里可以运行你的GPU任务
            print("\n现在开始运行你的GPU任务...")
            print("监控正在后台进行")
            
            # 模拟一些工作（在实际使用中，这里应该是你的GPU任务）
            print("按回车键停止监控并获取峰值显存用量...")
            input()  # 等待用户按回车
            
            # 停止监控并获取结果
            peak_memory_gb = monitor.stop()
            
            print(f"\n监控完成！峰值显存用量: {peak_memory_gb:.3f} GB")
    
    except Exception as e:
        print(f"错误: {e}")


# 多GPU监控示例
class MultiGPUMonitor:
    """多GPU监控器"""
    def __init__(self, gpu_indices=None, interval=1.0, unit='GB'):
        """
        初始化多GPU监控器
        
        Args:
            gpu_indices: 要监控的GPU索引列表，None表示监控所有GPU
            interval: 监控间隔
            unit: 返回的单位
        """
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        if gpu_indices is None:
            gpu_indices = list(range(device_count))
        
        self.monitors = []
        for idx in gpu_indices:
            if idx < device_count:
                monitor = GPUMemoryMonitor(gpu_index=idx, interval=interval, unit=unit)
                self.monitors.append(monitor)
            else:
                print(f"警告: GPU索引 {idx} 不存在，跳过")
    
    def start_all(self):
        """启动所有GPU监控"""
        results = []
        for monitor in self.monitors:
            success = monitor.start()
            results.append((monitor.gpu_index, success))
        return results
    
    def stop_all(self):
        """停止所有GPU监控并返回结果"""
        results = {}
        max_peak = 0.0
        for monitor in self.monitors:
            peak = monitor.stop()
            results[monitor.gpu_index] = {
                'name': monitor.gpu_name,
                'peak_memory': peak,
                'unit': monitor.unit
            }
            if peak > max_peak:
                max_peak = peak
        return results, max_peak
    
    def stop_all_and_get_max(self):
        """停止所有监控并返回最大峰值"""
        results = self.stop_all()
        if not results:
            return 0.0
        
        max_peak = max(item['peak_memory'] for item in results.values())
        return max_peak


# 快速使用函数
def quick_monitor(gpu_index=0, interval=0.5, unit='GB', wait_for_input=True):
    """
    快速启动监控的便捷函数
    
    Args:
        gpu_index: GPU索引
        interval: 监控间隔
        unit: 单位
        wait_for_input: 是否等待用户输入
    
    Returns:
        float: 峰值显存使用量
    """
    monitor = GPUMemoryMonitor(gpu_index=gpu_index, interval=interval, unit=unit)
    
    try:
        if monitor.start():
            if wait_for_input:
                print("按回车键停止监控...")
                input()
            else:
                # 如果不等待用户输入，这里可以设置其他停止条件
                # 例如：监控特定时间或直到某个条件满足
                print("监控已启动，将在后台运行")
                print("调用 monitor.stop() 来停止并获取结果")
                return monitor
        return None
    except Exception as e:
        print(f"监控失败: {e}")
        return None


# 测试代码
if __name__ == "__main__":
    # # 示例1: 基本使用
    # print("示例1: 基本使用")
    # monitor = GPUMemoryMonitor(gpu_index=0, interval=0.5, unit='GB')
    # monitor.start()
    
    # # 模拟一些GPU工作
    # print("模拟GPU工作...")
    # time.sleep(3)
    
    # # 停止并获取结果
    # peak = monitor.stop()
    # print(f"峰值显存: {peak:.3f} GB\n")
    
    # # 示例2: 使用便捷函数
    # print("示例2: 使用便捷函数")
    # result = quick_monitor(gpu_index=0, interval=0.2, unit='MB')
    # if result:
    #     # 这里 result 是 monitor 对象
    #     time.sleep(2)
    #     peak = result.stop()
    #     print(f"峰值显存: {peak:.3f} MB")
    
    # 示例3: 多GPU监控
    print("\n示例3: 多GPU监控")
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"系统中有 {device_count} 个GPU")
        
        if device_count > 1:
            # multi_monitor = MultiGPUMonitor(gpu_indices=[0, 1], interval=0.5, unit='GB')
            multi_monitor = MultiGPUMonitor(interval=5, unit='GB')
            multi_monitor.start_all()
            time.sleep(2)
            results, max_peak = multi_monitor.stop_all()
            
            for gpu_idx, data in results.items():
                print(f"GPU {gpu_idx} ({data['name']}): {data['peak_memory']:.3f} {data['unit']}")
            print(f"最大峰值显存使用量: {max_peak:.3f} GB")
    except:
        pass