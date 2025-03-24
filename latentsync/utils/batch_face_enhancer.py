import threading
from queue import Queue
import torch
import numpy as np
from .face_enhancer import FaceEnhancer

class BatchFaceEnhancer:
    def __init__(self, 
                 model_path: str,
                 num_workers: int = 2,
                 device: str = 'cuda',
                 enhancement_method: str = 'gfpgan',
                 enhancement_strength: float = 1.0,
                 mouth_protection: bool = True,
                 mouth_protection_strength: float = 0.8):
        """批处理人脸增强器
        
        Args:
            model_path: 模型路径
            num_workers: 工作线程数
            device: 设备类型 ('cuda' 或 'cpu')
            enhancement_method: 增强方法
            enhancement_strength: 增强强度 (0-1)
            mouth_protection: 是否保护嘴唇区域
            mouth_protection_strength: 嘴唇保护强度 (0-1)
        """
        self.num_workers = num_workers
        
        # 创建输入输出队列
        self.input_queue = Queue()
        self.output_queue = Queue()
        
        # 创建工作线程和增强器
        self.workers = []
        self.enhancers = []
        
        for i in range(num_workers):
            enhancer = FaceEnhancer(
                model_path=model_path,
                device=f"{device}:{i % torch.cuda.device_count()}" if device == 'cuda' else device,
                enhancement_method=enhancement_method,
                enhancement_strength=enhancement_strength,
                enable=True,
                mouth_protection=mouth_protection,
                mouth_protection_strength=mouth_protection_strength
            )
            self.enhancers.append(enhancer)
            
            worker = threading.Thread(target=self._worker_loop, args=(i, enhancer))
            worker.daemon = True
            self.workers.append(worker)
            worker.start()
        
        print(f"初始化完成，使用 {num_workers} 个工作线程")
        
    def _worker_loop(self, worker_id: int, enhancer: FaceEnhancer):
        """工作线程循环"""
        print(f"工作线程 {worker_id} 启动")
        while True:
            try:
                # 从输入队列获取任务
                idx, frame, landmarks = self.input_queue.get()
                if frame is None:  # 退出信号
                    break
                    
                # 处理人脸
                try:
                    enhanced = enhancer.enhance(frame, landmarks)
                    self.output_queue.put((idx, enhanced))
                except Exception as e:
                    print(f"工作线程 {worker_id} 处理帧 {idx} 时出错: {str(e)}")
                    self.output_queue.put((idx, frame))  # 出错时使用原始帧
                    
                # 标记任务完成
                self.input_queue.task_done()
                
            except Exception as e:
                print(f"工作线程 {worker_id} 出错: {str(e)}")
                continue
    
    def process_frame(self, frame_idx: int, frame: np.ndarray, landmarks=None):
        """提交一帧进行处理"""
        self.input_queue.put((frame_idx, frame, landmarks))
    
    def get_result(self, timeout=None):
        """获取处理结果"""
        try:
            return self.output_queue.get(timeout=timeout)
        except Exception as e:
            print(f"获取结果超时: {str(e)}")
            return None, None
    
    def __del__(self):
        """清理资源"""
        # 发送退出信号
        for _ in range(self.num_workers):
            self.input_queue.put((None, None, None))
        
        # 等待所有工作线程结束
        for worker in self.workers:
            worker.join(timeout=1.0)
            
        # 清理 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 