import os
import cv2
import numpy as np
from typing import Union, Optional, Tuple, List
import warnings
import onnxruntime
import torch

class FaceEnhancer:
    """人脸增强器，支持使用ONNX模型的GPEN，GFPGAN和CodeFormer三种增强方式"""
    
    def __init__(self, enhancement_method='gfpgan', model_path=None, device='cuda', enhancement_strength=0.5, 
                 enable=True, mouth_protection=False, mouth_protection_strength=0.8):
        """初始化人脸增强器
        
        Args:
            enhancement_method: 增强方法，支持'gpen', 'gfpgan'和'codeformer'
            model_path: ONNX模型路径，默认为None，会使用默认路径
            device: 设备，默认为'cuda'
            enhancement_strength: 增强强度，默认为0.5
            enable: 是否启用增强，默认为True
            mouth_protection: 是否保护嘴唇区域，默认为True
            mouth_protection_strength: 嘴唇保护强度，默认为0.8
        """
        self.enhancement_method = enhancement_method.lower()
        self.device = device
        self.enhancement_strength = enhancement_strength
        self.enable = enable
        self.mouth_protection = mouth_protection
        self.mouth_protection_strength = mouth_protection_strength
        
        # 设置默认模型路径
        if model_path is None:
            if self.enhancement_method == 'gfpgan':
                self.model_path = 'models/faceenhancer/GFPGANv1.4.onnx'
            elif self.enhancement_method == 'codeformer':
                self.model_path = 'models/faceenhancer/CodeFormerFixed.onnx'
            elif self.enhancement_method == 'gpen':
                self.model_path = 'models/faceenhancer/GPEN-BFR-512.onnx'
            else:
                self.model_path = None
        else:
            self.model_path = model_path
        
        # 初始化 ONNX 会话相关属性
        self.session = None
        self.io_binding = None
        self.input_name = None
        self.output_name = None
        self.resolution = None
        
        # 添加调试相关属性
        self.debug_dir = 'debug_frames'
        self.debug_count = 0
        self.max_debug_frames = 3
        
        # 创建调试目录
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
            print(f"创建调试目录: {self.debug_dir}")
        
        # 尝试加载ONNX模型
        if self.enable:
            self._load_onnx_model()
    
    def _load_onnx_model(self):
        """加载ONNX模型，使用优化的配置"""
        try:
            # 检查模型文件是否存在
            if not os.path.exists(self.model_path):
                print(f"ONNX模型不存在: {self.model_path}")
                print(f"请确保ONNX模型文件已放置到正确位置")
                self.enable = False
                return

            # 配置ONNX运行时选项
            session_options = onnxruntime.SessionOptions()
            
            # 启用所有图优化
            session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # 启用内存优化
            session_options.enable_mem_pattern = True
            session_options.enable_mem_reuse = True
            
            # 启用并行执行
            session_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
            
            # 配置线程数
            num_threads = min(os.cpu_count(), 4)  # 使用最多4个线程
            session_options.intra_op_num_threads = num_threads
            session_options.inter_op_num_threads = num_threads
            
            # 配置 CUDA Provider 选项
            provider_options = {
                "cudnn_conv_algo_search": "EXHAUSTIVE",  # 使用穷举搜索找到最快的卷积算法
                "device_id": 0,  # 使用第一个 GPU
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB GPU 内存限制
            }
            
            # 根据设备选择执行提供程序
            providers = []
            if str(self.device).lower() == 'cuda':
                if torch.cuda.is_available():  # 确保 CUDA 可用
                    providers = [
                        ("CUDAExecutionProvider", provider_options),
                        "CPUExecutionProvider"
                    ]
                else:
                    print("CUDA不可用，回退到CPU执行")
                    providers = ["CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
            
            # 创建ONNX会话
            self.session = onnxruntime.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers
            )
            
            # 获取并缓存模型信息
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # 获取输入尺寸
            input_shape = self.session.get_inputs()[0].shape
            if len(input_shape) >= 4:  # [batch, channels, height, width]
                self.resolution = (input_shape[-1], input_shape[-2])
            else:
                # 使用默认分辨率
                self.resolution = (512, 512)
            
            # 创建 IO Binding
            self.io_binding = self.session.io_binding()
            
            # 预热模型
            self._warmup_model()
            
            print(f"{self.enhancement_method.upper()} ONNX模型加载成功，输入尺寸: {self.resolution}")
            print(f"已启用优化配置：线程数={num_threads}，设备={self.device}")
            
        except Exception as e:
            print(f"加载ONNX模型失败: {str(e)}")
            self.session = None
            self.enable = False
    
    def _warmup_model(self):
        """预热模型，运行几次推理以确保 CUDA kernels 被编译和缓存"""
        try:
            # 创建随机输入数据
            dummy_input = np.random.randn(1, 3, *self.resolution).astype(np.float32)
            
            print(f"正在预热{self.enhancement_method.upper()}模型...")
            for i in range(3):  # 运行3次预热
                if self.enhancement_method == 'codeformer':
                    w = np.array([self.enhancement_strength], dtype=np.float64)
                    self.session.run(None, {
                        'x': dummy_input,
                        'w': w
                    })
                else:
                    self.session.run(None, {self.input_name: dummy_input})
                print(f"预热进度: {i+1}/3")
            print("模型预热完成")
            
        except Exception as e:
            print(f"模型预热失败: {str(e)}")
            print("继续使用未预热的模型")
    
    def preprocess(self, img):
        """预处理图像
        
        Args:
            img: 输入图像，BGR格式
            
        Returns:
            预处理后的图像和原始大小的图像
        """
        # 保存原始图像大小
        self.original_height, self.original_width = img.shape[:2]
        
        # 保存原始图像的副本以便后处理
        self.original_img = cv2.resize(img.copy(), self.resolution, interpolation=cv2.INTER_LINEAR)
        
        # 调整图像大小以匹配模型输入尺寸
        img_resized = cv2.resize(img, self.resolution, interpolation=cv2.INTER_LINEAR)
        
        # 转换为浮点型并标准化
        img_norm = img_resized.astype(np.float32)[:,:,::-1] / 255.0  # BGR -> RGB, normalize to [0, 1]
        
        # 转换为NCHW格式
        img_transposed = img_norm.transpose((2, 0, 1))  # HWC -> CHW
        
        # 标准化到[-1, 1]范围
        img_normalized = (img_transposed - 0.5) / 0.5
        
        # 添加批次维度
        img_batch = np.expand_dims(img_normalized, axis=0).astype(np.float32)
        
        return img_batch
    
    def postprocess(self, img):
        """后处理ONNX模型输出
        
        Args:
            img: 模型输出，NCHW格式，范围[-1, 1]
            
        Returns:
            处理后的BGR图像，范围[0, 255]，uint8类型
        """
        # 如果输出是4D张量(NCHW)，去掉批次维度
        if len(img.shape) == 4:
            img = img[0]
        
        # 从CHW转换回HWC
        img = img.transpose(1, 2, 0)
        
        # 从[-1, 1]转换回[0, 1]
        img = (img + 1) * 0.5
        
        # 从[0, 1]转换回[0, 255]并从RGB转换回BGR
        img = (img * 255.0)[:, :, ::-1]
        
        # 裁剪到[0, 255]范围并转换为uint8
        img = np.clip(img, 0, 255).astype('uint8')
        
        # 调整回原始图像大小
        img = cv2.resize(img, (self.original_width, self.original_height), interpolation=cv2.INTER_LANCZOS4)
        
        return img
    
    def enhance(self, img: np.ndarray, face_landmarks=None) -> np.ndarray:
        """增强图像
        
        Args:
            img: 输入图像，BGR格式
            face_landmarks: 人脸关键点，用于保护嘴唇区域，可选
            
        Returns:
            增强后的图像
        """
        # 如果未启用或模型为空，返回原图
        if not self.enable or self.session is None:
            return img
        
        # 拷贝图像以避免修改原图
        img = img.copy()
        
        try:
            # 保存调试图像（前3帧）
            if self.debug_count < self.max_debug_frames:
                debug_prefix = f"{self.enhancement_method}_frame_{self.debug_count}"
                # 保存原始图像
                cv2.imwrite(os.path.join(self.debug_dir, f"{debug_prefix}_before.png"), img)
                print(f"保存原始图像: {debug_prefix}_before.png")
            
            # 预处理图像
            input_data = self.preprocess(img)
            
            # 获取模型输出形状
            output_shape = self.session.get_outputs()[0].shape
            if output_shape[0] == -1:  # 如果批次维度是动态的
                output_shape = list(input_data.shape)
            
            # 运行推理
            if self.enhancement_method == 'codeformer':
                # CodeFormer需要两个输入
                w = np.array([self.enhancement_strength], dtype=np.float64)
                output = self.session.run(None, {
                    'x': input_data,
                    'w': w
                })[0]
            else:
                output = self.session.run(None, {self.input_name: input_data})[0]
            
            # 后处理输出
            enhanced_img = self.postprocess(output)
            
            # 使用 CUDA 加速图像混合
            if self.enhancement_strength < 1.0:
                # 调整原图大小以匹配增强结果
                original_resized = cv2.resize(img, (enhanced_img.shape[1], enhanced_img.shape[0]), 
                                           interpolation=cv2.INTER_LINEAR)
                
                if str(self.device).lower() == 'cuda' and torch.cuda.is_available():
                    # 使用 CUDA 进行图像混合
                    enhanced_tensor = torch.from_numpy(enhanced_img).cuda()
                    original_tensor = torch.from_numpy(original_resized).cuda()
                    
                    enhanced_tensor = enhanced_tensor * self.enhancement_strength + \
                                    original_tensor * (1.0 - self.enhancement_strength)
                    
                    enhanced_img = enhanced_tensor.cpu().numpy()
                else:
                    # CPU 混合
                    enhanced_img = cv2.addWeighted(enhanced_img, self.enhancement_strength,
                                                 original_resized, 1.0 - self.enhancement_strength, 0)
            
            # 优化嘴唇保护处理
            if self.mouth_protection and face_landmarks is not None and len(face_landmarks) > 0:
                enhanced_img = self._apply_mouth_protection(enhanced_img, img, face_landmarks)
            
            # 清理 CUDA 缓存
            if str(self.device).lower() == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 保存增强后的图像（前3帧）
            if self.debug_count < self.max_debug_frames:
                # 保存增强后的图像
                cv2.imwrite(os.path.join(self.debug_dir, f"{debug_prefix}_after.png"), enhanced_img)
                print(f"保存增强后图像: {debug_prefix}_after.png")
                
                # 如果有嘴唇保护，也保存嘴唇遮罩
                if self.mouth_protection and face_landmarks is not None and len(face_landmarks) > 0:
                    mask = np.zeros_like(enhanced_img)
                    for landmarks in face_landmarks:
                        mouth_points = self._get_mouth_points(landmarks, enhanced_img.shape)
                        cv2.fillPoly(mask, [np.array(mouth_points, dtype=np.int32)], (255, 255, 255))
                    cv2.imwrite(os.path.join(self.debug_dir, f"{debug_prefix}_mouth_mask.png"), mask)
                    print(f"保存嘴唇遮罩: {debug_prefix}_mouth_mask.png")
                
                self.debug_count += 1
            
            return enhanced_img
            
        except Exception as e:
            print(f"面部增强过程出错: {str(e)}")
            return img
            
    def _apply_mouth_protection(self, enhanced_img, original_img, face_landmarks):
        """应用嘴唇保护
        
        Args:
            enhanced_img: 增强后的图像
            original_img: 原始图像
            face_landmarks: 人脸关键点
            
        Returns:
            处理后的图像
        """
        try:
            # 创建嘴唇区域遮罩
            mask = np.zeros_like(enhanced_img)
            
            # 对每个人脸处理嘴唇区域
            for landmarks in face_landmarks:
                mouth_points = self._get_mouth_points(landmarks, enhanced_img.shape)
                
                # 创建嘴唇区域多边形
                cv2.fillPoly(mask, [np.array(mouth_points, dtype=np.int32)], (255, 255, 255))
                
                # 扩大嘴唇区域
                kernel = np.ones((15, 15), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
            
            # 缩放原始图像以匹配增强后的图像大小
            original_resized = cv2.resize(original_img, 
                                        (enhanced_img.shape[1], enhanced_img.shape[0]),
                                        interpolation=cv2.INTER_LINEAR)
            
            if str(self.device).lower() == 'cuda' and torch.cuda.is_available():
                # 使用 CUDA 进行图像混合
                mask_tensor = torch.from_numpy(mask).cuda().float() / 255.0
                enhanced_tensor = torch.from_numpy(enhanced_img).cuda()
                original_tensor = torch.from_numpy(original_resized).cuda()
                
                if self.mouth_protection_strength < 1.0:
                    # 部分保护
                    mask_strength = mask_tensor * self.mouth_protection_strength
                    result = (1 - mask_strength) * enhanced_tensor + mask_strength * original_tensor
                else:
                    # 完全保护
                    result = torch.where(mask_tensor > 0, original_tensor, enhanced_tensor)
                
                enhanced_img = result.cpu().numpy()
            else:
                # CPU 处理
                if self.mouth_protection_strength < 1.0:
                    # 部分保护
                    mask_float = mask.astype(np.float32) / 255.0
                    mask_strength = mask_float * self.mouth_protection_strength
                    for c in range(3):
                        enhanced_img[:,:,c] = (1 - mask_strength[:,:,c]) * enhanced_img[:,:,c] + \
                                             mask_strength[:,:,c] * original_resized[:,:,c]
                else:
                    # 完全保护
                    mask = mask.astype(bool)
                    enhanced_img[mask] = original_resized[mask]
            
            return enhanced_img
            
        except Exception as e:
            print(f"应用嘴唇保护时出错: {str(e)}")
            return enhanced_img
            
    def _get_mouth_points(self, landmarks, img_shape):
        """获取嘴唇关键点
        
        Args:
            landmarks: 人脸关键点
            img_shape: 图像形状
            
        Returns:
            嘴唇区域的关键点
        """
        try:
            # 对于68点模型，嘴唇点是48-68
            if len(landmarks) >= 68:
                return landmarks[48:68]
            # 如果是简化的关键点模型
            elif len(landmarks) >= 20:
                return landmarks[-8:]  # 取最后8个点作为嘴唇
            else:
                # 使用固定区域
                h, w = img_shape[:2]
                return np.array([
                    [w//2 - w//8, h//2 + h//8],
                    [w//2 + w//8, h//2 + h//8],
                    [w//2 + w//8, h//2 + h//4],
                    [w//2 - w//8, h//2 + h//4]
                ])
        except:
            # 如果出错，使用固定区域
            h, w = img_shape[:2]
            return np.array([
                [w//2 - w//8, h//2 + h//8],
                [w//2 + w//8, h//2 + h//8],
                [w//2 + w//8, h//2 + h//4],
                [w//2 - w//8, h//2 + h//4]
            ]) 