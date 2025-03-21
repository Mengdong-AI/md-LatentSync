import os
import cv2
import numpy as np
from typing import Union, Optional, Tuple, List
import warnings
import onnxruntime
import time
import traceback

class FaceEnhancer:
    """人脸增强器，支持使用ONNX模型的GPEN，GFPGAN和CodeFormer三种增强方式"""
    
    def __init__(self, enhancement_method='gfpgan', model_path=None, device='cuda', enhancement_strength=0.5, 
                 enable=True, mouth_protection=True, mouth_protection_strength=0.8):
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
                self.model_path = 'models/faceenhancer/codeformer.onnx'
            elif self.enhancement_method == 'gpen':
                self.model_path = 'models/faceenhancer/GPEN-BFR-512.onnx'
            else:
                self.model_path = None
                print(f"[初始化] 错误: 不支持的增强方法: {self.enhancement_method}")
                self.enable = False
        else:
            self.model_path = model_path
        
        # 尝试加载ONNX模型
        self.session = None
        if self.enable:
            self._load_onnx_model()
    
    def _load_onnx_model(self):
        """加载ONNX模型"""
        try:
            # 检查模型文件是否存在
            if not os.path.exists(self.model_path):
                print(f"[加载] 错误: ONNX模型不存在: {self.model_path}")
                print(f"[加载] 当前工作目录: {os.getcwd()}")
                print(f"[加载] 请确保ONNX模型文件已放置到正确位置")
                
                # 检查模型目录是否存在
                model_dir = os.path.dirname(self.model_path)
                if not os.path.exists(model_dir):
                    print(f"[加载] 错误: 模型目录不存在: {model_dir}")
                    try:
                        os.makedirs(model_dir, exist_ok=True)
                        print(f"[加载] 已创建模型目录: {model_dir}")
                    except Exception as e:
                        print(f"[加载] 创建模型目录失败: {str(e)}")
                else:
                    print(f"[加载] 模型目录存在，但模型文件不存在")
                
                self.enable = False
                return

            # 配置ONNX运行时选项
            session_options = onnxruntime.SessionOptions()
            session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # 根据设备选择执行提供程序
            providers = ["CPUExecutionProvider"]
            if self.device.lower() == 'cuda':
                if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
                    providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
                    print(f"[加载] 使用CUDA加速")
                else:
                    print(f"[加载] 警告: CUDA不可用，回退到CPU")
            
            print(f"[加载] 可用的ONNX提供程序: {onnxruntime.get_available_providers()}")
            print(f"[加载] 使用的ONNX提供程序: {providers}")
            
            # 创建ONNX会话
            self.session = onnxruntime.InferenceSession(
                self.model_path, 
                sess_options=session_options, 
                providers=providers
            )
            
            # 获取模型的输入尺寸
            self.input_name = self.session.get_inputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            if len(input_shape) >= 4:  # [batch, channels, height, width]
                self.resolution = (input_shape[-1], input_shape[-2])
            else:
                # 使用默认分辨率
                self.resolution = (512, 512)
                
            print(f"[加载] {self.enhancement_method.upper()} ONNX模型加载成功，输入尺寸: {self.resolution}")
            print(f"[加载] 模型输入名称: {self.input_name}, 模型输入形状: {input_shape}")
            print(f"[加载] 模型输出形状: {self.session.get_outputs()[0].shape}")
            
        except Exception as e:
            print(f"[加载] 加载ONNX模型失败: {str(e)}")
            traceback.print_exc()
            self.session = None
            self.enable = False
    
    def preprocess(self, img):
        """预处理图像
        
        Args:
            img: 输入图像，BGR格式
            
        Returns:
            预处理后的图像和原始大小的图像
        """
        print(f"[预处理] 输入图像 - 形状: {img.shape}, 类型: {img.dtype}, 范围: [{np.min(img)}, {np.max(img)}]")
        if img.size == 0 or np.isnan(img).any():
            print(f"[预处理] 警告: 输入图像包含无效值(NaN)或为空")
        
        # 保存原始图像大小
        self.original_height, self.original_width = img.shape[:2]
        print(f"[预处理] 原始图像大小: {self.original_width}x{self.original_height}")
        
        # 保存原始图像的副本以便后处理
        self.original_img = cv2.resize(img.copy(), self.resolution, interpolation=cv2.INTER_LINEAR)
        
        # 调整图像大小以匹配模型输入尺寸
        img_resized = cv2.resize(img, self.resolution, interpolation=cv2.INTER_LINEAR)
        print(f"[预处理] 调整大小后 - 形状: {img_resized.shape}, 类型: {img_resized.dtype}")
        
        # 转换为浮点型并标准化
        img_norm = img_resized.astype(np.float32)[:,:,::-1] / 255.0  # BGR -> RGB, normalize to [0, 1]
        print(f"[预处理] BGR->RGB标准化后 - 形状: {img_norm.shape}, 类型: {img_norm.dtype}, 范围: [{np.min(img_norm)}, {np.max(img_norm)}]")
        
        # 转换为NCHW格式
        img_transposed = img_norm.transpose((2, 0, 1))  # HWC -> CHW
        print(f"[预处理] HWC->CHW后 - 形状: {img_transposed.shape}, 类型: {img_transposed.dtype}")
        
        # 标准化到[-1, 1]范围
        img_normalized = (img_transposed - 0.5) / 0.5
        print(f"[预处理] 标准化到[-1,1]后 - 形状: {img_normalized.shape}, 类型: {img_normalized.dtype}, 范围: [{np.min(img_normalized)}, {np.max(img_normalized)}]")
        
        # 添加批次维度
        img_batch = np.expand_dims(img_normalized, axis=0).astype(np.float32)
        print(f"[预处理] 最终输入 - 形状: {img_batch.shape}, 类型: {img_batch.dtype}, 范围: [{np.min(img_batch)}, {np.max(img_batch)}]")
        
        return img_batch
    
    def postprocess(self, img):
        """后处理ONNX模型输出
        
        Args:
            img: 模型输出，NCHW格式，范围[-1, 1]
            
        Returns:
            处理后的BGR图像，范围[0, 255]，uint8类型
        """
        try:
            print(f"[后处理] 原始输出 - 形状: {img.shape}, 类型: {img.dtype}, 范围: [{np.min(img)}, {np.max(img)}]")
            if img is None:
                print(f"[后处理] 错误: 输入为None")
                return np.zeros((self.original_height, self.original_width, 3), dtype=np.uint8)
                
            if np.isnan(img).any():
                print(f"[后处理] 警告: 输入包含NaN值")
                
            # 如果输出是4D张量(NCHW)，去掉批次维度
            if len(img.shape) == 4:
                img = img[0]
                print(f"[后处理] 去除批次维度后 - 形状: {img.shape}")
            
            # 确保图像是32位浮点数类型，避免CV_16F类型不兼容的问题
            if img.dtype != np.float32:
                print(f"[后处理] 转换数据类型: {img.dtype} -> float32")
                img = img.astype(np.float32)
            
            # 从CHW转换回HWC
            img = img.transpose(1, 2, 0)
            print(f"[后处理] CHW->HWC后 - 形状: {img.shape}, 类型: {img.dtype}")
            
            # 从[-1, 1]转换回[0, 1]
            img = (img + 1) * 0.5
            print(f"[后处理] [-1,1]->[0,1]后 - 范围: [{np.min(img)}, {np.max(img)}]")
            
            # 从[0, 1]转换回[0, 255]并从RGB转换回BGR
            img = (img * 255.0)[:, :, ::-1]
            print(f"[后处理] RGB->BGR并缩放到[0,255]后 - 范围: [{np.min(img)}, {np.max(img)}]")
            
            # 裁剪到[0, 255]范围并转换为uint8
            img = np.clip(img, 0, 255).astype('uint8')
            print(f"[后处理] 裁剪并转为uint8后 - 形状: {img.shape}, 类型: {img.dtype}, 范围: [{np.min(img)}, {np.max(img)}]")
            
            # 调整回原始图像大小
            img = cv2.resize(img, (self.original_width, self.original_height), interpolation=cv2.INTER_LANCZOS4)
            print(f"[后处理] 调整回原始大小后 - 形状: {img.shape}, 类型: {img.dtype}")
            
            return img
        except Exception as e:
            print(f"[后处理] 错误: {str(e)}")
            print(f"[后处理] 输入图像形状: {img.shape if hasattr(img, 'shape') else 'Unknown'}, 类型: {img.dtype if hasattr(img, 'dtype') else 'Unknown'}")
            # 返回空图像
            return np.zeros((self.original_height, self.original_width, 3), dtype=np.uint8)
    
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
            print(f"[增强] 跳过: 增强未启用或模型未加载")
            return img
            
        # 检查输入图像
        print(f"[增强] 输入图像 - 形状: {img.shape}, 类型: {img.dtype}, 非零值数量: {np.count_nonzero(img)}")
        if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            print(f"[增强] 错误: 输入图像尺寸无效")
            return img
            
        if np.count_nonzero(img) == 0:
            print(f"[增强] 警告: 输入图像全为零")
            
        # 拷贝图像以避免修改原图
        img = img.copy()
        
        try:
            # 预处理图像
            print(f"[增强] 开始预处理图像...")
            input_data = self.preprocess(img)
            
            # 运行ONNX推理
            try:
                print(f"[增强] 运行ONNX推理...")
                start_time = time.time() if 'time' in globals() else None
                output = self.session.run(None, {self.input_name: input_data})[0]
                if start_time:
                    print(f"[增强] ONNX推理完成, 耗时: {time.time() - start_time:.2f}秒")
                print(f"[增强] ONNX输出 - 形状: {output.shape}, 类型: {output.dtype}, 范围: [{np.min(output)}, {np.max(output)}]")
                
                # 检查输出是否有效
                if np.isnan(output).any():
                    print(f"[增强] 警告: 模型输出包含NaN值")
                if np.isinf(output).any():
                    print(f"[增强] 警告: 模型输出包含无穷值")
            except Exception as e:
                print(f"[增强] ONNX推理失败: {str(e)}")
                return img
            
            # 检查输出数据类型，确保是float32
            if output.dtype != np.float32:
                print(f"[增强] 警告: 模型输出数据类型不是float32，而是{output.dtype}，尝试转换")
                try:
                    output = output.astype(np.float32)
                except Exception as e:
                    print(f"[增强] 转换数据类型失败: {str(e)}，返回原始图像")
                    return img
            
            # 后处理输出
            print(f"[增强] 开始后处理图像...")
            enhanced_img = self.postprocess(output)
            print(f"[增强] 后处理完成 - 形状: {enhanced_img.shape}, 类型: {enhanced_img.dtype}, 范围: [{np.min(enhanced_img)}, {np.max(enhanced_img)}]")
            
            # 根据增强强度混合原图和增强结果
            if self.enhancement_strength < 1.0:
                print(f"[增强] 应用增强强度: {self.enhancement_strength}")
                # 调整原图大小以匹配增强结果
                original_resized = cv2.resize(img, (enhanced_img.shape[1], enhanced_img.shape[0]), 
                                             interpolation=cv2.INTER_LANCZOS4)
                enhanced_img = cv2.addWeighted(enhanced_img, self.enhancement_strength, 
                                              original_resized, 1.0 - self.enhancement_strength, 0)
            
            # 如果启用嘴唇保护并有人脸关键点，保留原始嘴唇区域
            if self.mouth_protection and face_landmarks is not None and len(face_landmarks) > 0:
                print(f"[增强] 应用嘴唇保护, 保护强度: {self.mouth_protection_strength}")
                try:
                    # 创建嘴唇区域遮罩
                    mask = np.zeros_like(enhanced_img)
                    
                    # 对每个人脸
                    for landmarks in face_landmarks:
                        # 获取嘴唇关键点（通常是最后的点）
                        try:
                            # 对于68点模型，嘴唇点是48-68
                            if len(landmarks) >= 68:
                                mouth_points = landmarks[48:68]
                            # 如果是简化的关键点模型
                            elif len(landmarks) >= 20:
                                mouth_points = landmarks[-8:]  # 取最后8个点作为嘴唇
                            else:
                                # 使用固定区域
                                h, w = enhanced_img.shape[:2]
                                mouth_points = np.array([
                                    [w//2 - w//8, h//2 + h//8],
                                    [w//2 + w//8, h//2 + h//8],
                                    [w//2 + w//8, h//2 + h//4],
                                    [w//2 - w//8, h//2 + h//4]
                                ])
                        except:
                            # 如果出错，使用固定区域
                            h, w = enhanced_img.shape[:2]
                            mouth_points = np.array([
                                [w//2 - w//8, h//2 + h//8],
                                [w//2 + w//8, h//2 + h//8],
                                [w//2 + w//8, h//2 + h//4],
                                [w//2 - w//8, h//2 + h//4]
                            ])
                        
                        # 创建嘴唇区域多边形
                        cv2.fillPoly(mask, [np.array(mouth_points, dtype=np.int32)], (255, 255, 255))
                        
                        # 扩大嘴唇区域
                        kernel = np.ones((15, 15), np.uint8)
                        mask = cv2.dilate(mask, kernel, iterations=1)
                    
                    # 应用遮罩，保留原始嘴唇区域
                    # 缩放原始图像以匹配增强后的图像大小
                    original_resized = cv2.resize(img, (enhanced_img.shape[1], enhanced_img.shape[0]), 
                                                interpolation=cv2.INTER_LANCZOS4)
                    
                    # 创建嘴唇保护混合
                    if self.mouth_protection_strength < 1.0:
                        # 部分保护，混合原始和增强的嘴唇区域
                        mask_float = mask.astype(np.float32) / 255.0
                        mask_strength = mask_float * self.mouth_protection_strength
                        for c in range(3):
                            enhanced_img[:,:,c] = (1 - mask_strength[:,:,c]) * enhanced_img[:,:,c] + \
                                                 mask_strength[:,:,c] * original_resized[:,:,c]
                    else:
                        # 完全保护，直接替换
                        mask = mask.astype(bool)
                        enhanced_img[mask] = original_resized[mask]
                
                except Exception as e:
                    print(f"应用嘴唇保护时出错: {str(e)}")
            
            return enhanced_img
            
        except Exception as e:
            print(f"面部增强过程出错: {str(e)}")
            return img 