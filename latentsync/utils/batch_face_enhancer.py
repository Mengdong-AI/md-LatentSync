import os
import time
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple
import numpy as np
import torch
import cv2
from .face_enhancer import FaceEnhancer

class BatchFaceEnhancer:
    def __init__(self, 
                 model_path: str,
                 batch_size: int = 4,
                 num_workers: int = 2,
                 queue_size: int = 8,
                 device: str = 'cuda',
                 enhancement_method: str = 'gfpgan',
                 enhancement_strength: float = 1.0,
                 mouth_protection: bool = True,
                 mouth_protection_strength: float = 0.8):
        """批处理人脸增强器
        
        Args:
            model_path: 模型路径
            batch_size: 批处理大小
            num_workers: 工作线程数
            queue_size: 队列大小
            device: 设备类型 ('cuda' 或 'cpu')
            enhancement_method: 增强方法 ('gfpgan' 或 'codeformer')
            enhancement_strength: 增强强度 (0-1)
            mouth_protection: 是否保护嘴唇区域
            mouth_protection_strength: 嘴唇保护强度 (0-1)
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.queue_size = queue_size
        
        # 创建输入输出队列
        self.input_queue = Queue(maxsize=queue_size)
        self.output_queue = Queue(maxsize=queue_size)
        
        # 创建工作线程池
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        # 创建多个 FaceEnhancer 实例用于并行处理
        self.enhancers = [
            FaceEnhancer(
                model_path=model_path,
                device=device,
                enhancement_method=enhancement_method,
                enhancement_strength=enhancement_strength,
                enable=True,
                mouth_protection=mouth_protection,
                mouth_protection_strength=mouth_protection_strength
            )
            for _ in range(num_workers)
        ]
        
        # 启动处理线程
        self.running = True
        self.process_thread = threading.Thread(target=self._process_batches)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # 添加性能监控
        self.metrics = {
            'processed_frames': 0,
            'processing_time': [],
            'queue_wait_time': [],
            'batch_sizes': []
        }
    
    def _process_batches(self):
        """批处理主循环"""
        print("启动批处理线程...")
        batch_count = 0
        
        while self.running:
            try:
                # 收集一个批次的数据
                batch_data = []
                batch_landmarks = []
                batch_indices = []
                
                # 记录等待时间
                start_wait = time.time()
                
                # 添加超时机制
                timeout_count = 0
                max_timeout = 10  # 最大等待10秒
                
                while len(batch_data) < self.batch_size and timeout_count < max_timeout:
                    try:
                        if not self.input_queue.empty():
                            idx, frame, landmarks = self.input_queue.get(timeout=1.0)
                            batch_data.append(frame)
                            batch_landmarks.append(landmarks)
                            batch_indices.append(idx)
                            timeout_count = 0  # 重置超时计数
                        else:
                            timeout_count += 1
                            time.sleep(0.1)  # 短暂等待
                    except Exception as e:
                        print(f"收集批次数据时出错: {str(e)}")
                        timeout_count += 1
                        continue
                
                if not batch_data:
                    continue
                
                batch_count += 1
                print(f"\n开始处理第 {batch_count} 批数据，包含 {len(batch_data)} 帧...")
                
                # 记录队列等待时间
                wait_time = time.time() - start_wait
                self.metrics['queue_wait_time'].append(wait_time)
                self.metrics['batch_sizes'].append(len(batch_data))
                
                # 并行处理批次数据
                start_process = time.time()
                futures = []
                
                for i in range(0, len(batch_data), self.num_workers):
                    worker_batch = batch_data[i:i + self.num_workers]
                    worker_landmarks = batch_landmarks[i:i + self.num_workers]
                    worker_indices = batch_indices[i:i + self.num_workers]
                    
                    for j, (frame, landmarks, idx) in enumerate(zip(worker_batch, worker_landmarks, worker_indices)):
                        try:
                            enhancer = self.enhancers[j]
                            future = self.executor.submit(enhancer.enhance, frame, landmarks)
                            futures.append((future, idx))
                            print(f"提交帧 {idx} 到工作线程 {j}")
                        except Exception as e:
                            print(f"提交帧 {idx} 到工作线程时出错: {str(e)}")
                            # 使用原始帧作为结果
                            self.output_queue.put((idx, frame))
                
                # 收集结果并按顺序放入输出队列
                successful_frames = 0
                for future, idx in futures:
                    try:
                        result = future.result(timeout=30)  # 设置30秒超时
                        self.output_queue.put((idx, result))
                        successful_frames += 1
                        self.metrics['processed_frames'] += 1
                        print(f"完成处理帧 {idx}")
                    except Exception as e:
                        print(f"处理帧 {idx} 时出错: {str(e)}")
                        # 使用原始帧作为结果
                        self.output_queue.put((idx, batch_data[batch_indices.index(idx)]))
                
                # 记录处理时间
                process_time = time.time() - start_process
                self.metrics['processing_time'].append(process_time)
                
                print(f"第 {batch_count} 批处理完成，成功处理 {successful_frames}/{len(batch_data)} 帧")
                print(f"处理时间: {process_time:.2f}秒，等待时间: {wait_time:.2f}秒")
                
                # 保持指标列表的大小
                max_metrics = 100
                for key in ['processing_time', 'queue_wait_time', 'batch_sizes']:
                    if len(self.metrics[key]) > max_metrics:
                        self.metrics[key] = self.metrics[key][-max_metrics:]
                
                # 清理CUDA缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"批处理循环出错: {str(e)}")
                # 短暂等待后继续
                time.sleep(1)
                continue
        
        print("批处理线程结束")
    
    def process_frame(self, frame_idx: int, frame: np.ndarray, landmarks=None) -> None:
        """提交一帧进行处理
        
        Args:
            frame_idx: 帧索引
            frame: 输入帧
            landmarks: 人脸关键点
        """
        self.input_queue.put((frame_idx, frame, landmarks))
    
    def get_result(self, timeout: Optional[float] = None) -> Tuple[int, np.ndarray]:
        """获取处理结果
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            (frame_idx, enhanced_frame): 帧索引和增强后的帧
        """
        return self.output_queue.get(timeout=timeout)
    
    def get_metrics(self):
        """获取性能指标"""
        if not self.metrics['processing_time']:
            return None
            
        return {
            'processed_frames': self.metrics['processed_frames'],
            'avg_processing_time': np.mean(self.metrics['processing_time']),
            'avg_queue_wait_time': np.mean(self.metrics['queue_wait_time']),
            'avg_batch_size': np.mean(self.metrics['batch_sizes']),
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize()
        }
    
    def __del__(self):
        """清理资源"""
        self.running = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        if hasattr(self, 'executor'):
            self.executor.shutdown() 