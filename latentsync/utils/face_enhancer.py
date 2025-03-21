import os
import cv2
import numpy as np
from typing import Union, Optional, Tuple, List
import warnings
import onnxruntime

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
                self.model_path = 'models/faceenhancer/CodeFormerFixed.onnx'
            elif self.enhancement_method == 'gpen':
                self.model_path = 'models/faceenhancer/GPEN-BFR-512.onnx'
            else:
                self.model_path = None
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
                print(f"ONNX模型不存在: {self.model_path}")
                print(f"请确保ONNX模型文件已放置到正确位置")
                self.enable = False
                return

            # 配置ONNX运行时选项
            session_options = onnxruntime.SessionOptions()
            session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # 根据设备选择执行提供程序
            providers = ["CPUExecutionProvider"]
            if str(self.device).lower() == 'cuda':
                providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
            
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
                
            print(f"{self.enhancement_method.upper()} ONNX模型加载成功，输入尺寸: {self.resolution}")
            
        except Exception as e:
            print(f"加载ONNX模型失败: {str(e)}")
            self.session = None
            self.enable = False
    
    def preprocess(self, img):
        """预处理图像
        
        Args:
            img: 输入图像，BGR格式
            
        Returns:
            预处理后的图像和原始大小的图像
        """
        try:
            print(f"[DEBUG] 预处理开始，输入图像形状: {img.shape}, 类型: {img.dtype}")
            
            # 保存原始图像大小
            self.original_height, self.original_width = img.shape[:2]
            
            try:
                # 保存原始图像的副本以便后处理
                self.original_img = cv2.resize(img.copy(), self.resolution, interpolation=cv2.INTER_LINEAR)
                print(f"[DEBUG] 调整大小后的原始图像形状: {self.original_img.shape}")
            except Exception as e:
                print(f"[ERROR] 调整原始图像大小失败: {str(e)}")
                print(f"[ERROR] 错误类型: {type(e)}")
                import traceback
                print(f"[ERROR] 堆栈跟踪:\n{traceback.format_exc()}")
                raise
            
            try:
                # 调整图像大小以匹配模型输入尺寸
                img_resized = cv2.resize(img, self.resolution, interpolation=cv2.INTER_LINEAR)
                print(f"[DEBUG] 调整大小后的图像形状: {img_resized.shape}")
            except Exception as e:
                print(f"[ERROR] 调整输入图像大小失败: {str(e)}")
                print(f"[ERROR] 错误类型: {type(e)}")
                import traceback
                print(f"[ERROR] 堆栈跟踪:\n{traceback.format_exc()}")
                raise
            
            try:
                # 转换为浮点型并标准化
                img_norm = img_resized.astype(np.float32)
                print(f"[DEBUG] 转换为float32后类型: {img_norm.dtype}")
                
                # BGR到RGB转换
                img_norm = img_norm[:,:,::-1]
                print(f"[DEBUG] BGR到RGB转换后形状: {img_norm.shape}")
                
                # 归一化到[0,1]
                img_norm = img_norm / 255.0
                print(f"[DEBUG] 标准化后的图像范围: [{img_norm.min()}, {img_norm.max()}]")
            except Exception as e:
                print(f"[ERROR] 颜色空间转换或归一化失败: {str(e)}")
                print(f"[ERROR] 错误类型: {type(e)}")
                import traceback
                print(f"[ERROR] 堆栈跟踪:\n{traceback.format_exc()}")
                raise
            
            try:
                # 转换为NCHW格式
                img_transposed = img_norm.transpose((2, 0, 1))  # HWC -> CHW
                print(f"[DEBUG] 转置后的图像形状: {img_transposed.shape}")
            except Exception as e:
                print(f"[ERROR] 转置操作失败: {str(e)}")
                print(f"[ERROR] 错误类型: {type(e)}")
                import traceback
                print(f"[ERROR] 堆栈跟踪:\n{traceback.format_exc()}")
                raise
            
            try:
                # 标准化到[-1, 1]范围
                img_normalized = (img_transposed - 0.5) / 0.5
                print(f"[DEBUG] 最终标准化后的范围: [{img_normalized.min()}, {img_normalized.max()}]")
            except Exception as e:
                print(f"[ERROR] 标准化到[-1,1]范围失败: {str(e)}")
                print(f"[ERROR] 错误类型: {type(e)}")
                import traceback
                print(f"[ERROR] 堆栈跟踪:\n{traceback.format_exc()}")
                raise
            
            try:
                # 添加批次维度
                img_batch = np.expand_dims(img_normalized, axis=0).astype(np.float32)
                print(f"[DEBUG] 最终预处理输出形状: {img_batch.shape}, 类型: {img_batch.dtype}")
            except Exception as e:
                print(f"[ERROR] 添加批次维度失败: {str(e)}")
                print(f"[ERROR] 错误类型: {type(e)}")
                import traceback
                print(f"[ERROR] 堆栈跟踪:\n{traceback.format_exc()}")
                raise
            
            return img_batch
            
        except Exception as e:
            print(f"[ERROR] 预处理过程出错: {str(e)}")
            print(f"[ERROR] 错误类型: {type(e)}")
            import traceback
            print(f"[ERROR] 完整错误堆栈:\n{traceback.format_exc()}")
            raise
    
    def postprocess(self, img):
        """后处理ONNX模型输出
        
        Args:
            img: 模型输出，NCHW格式，范围[-1, 1]
            
        Returns:
            处理后的BGR图像，范围[0, 255]，uint8类型
        """
        try:
            print(f"[DEBUG] 后处理开始，输入数据形状: {img.shape}, 类型: {img.dtype}, 值范围: [{img.min()}, {img.max()}]")
            
            # 确保数据类型是 float32
            if img.dtype != np.float32:
                print(f"[DEBUG] 将数据类型从 {img.dtype} 转换为 float32")
                img = img.astype(np.float32)
            
            try:
                # 从 NCHW 转换为 HWC
                img = np.squeeze(img)  # 移除批次维度
                img = img.transpose(1, 2, 0)  # CHW -> HWC
                print(f"[DEBUG] 转置后形状: {img.shape}, 类型: {img.dtype}")
            except Exception as e:
                print(f"[ERROR] 转置操作失败: {str(e)}")
                import traceback
                print(f"[ERROR] 堆栈跟踪:\n{traceback.format_exc()}")
                raise
            
            try:
                # 从 [-1, 1] 转换到 [0, 1]
                img = (img + 1) / 2
                img = np.clip(img, 0, 1)
                print(f"[DEBUG] 值范围调整后: [{img.min()}, {img.max()}]")
            except Exception as e:
                print(f"[ERROR] 值范围调整失败: {str(e)}")
                import traceback
                print(f"[ERROR] 堆栈跟踪:\n{traceback.format_exc()}")
                raise
            
            try:
                # 从 RGB 转换到 BGR
                img = img[:, :, ::-1]
                print(f"[DEBUG] RGB到BGR转换后形状: {img.shape}")
            except Exception as e:
                print(f"[ERROR] 颜色空间转换失败: {str(e)}")
                import traceback
                print(f"[ERROR] 堆栈跟踪:\n{traceback.format_exc()}")
                raise
            
            try:
                # 转换到 uint8 类型
                img = (img * 255).round()
                img = img.astype(np.uint8)
                print(f"[DEBUG] 最终输出形状: {img.shape}, 类型: {img.dtype}, 值范围: [{img.min()}, {img.max()}]")
            except Exception as e:
                print(f"[ERROR] 类型转换失败: {str(e)}")
                import traceback
                print(f"[ERROR] 堆栈跟踪:\n{traceback.format_exc()}")
                raise
            
            return img
            
        except Exception as e:
            print(f"[ERROR] 后处理过程出错: {str(e)}")
            print(f"[ERROR] 错误类型: {type(e)}")
            import traceback
            print(f"[ERROR] 完整错误堆栈:\n{traceback.format_exc()}")
            raise
    
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
            print("[DEBUG] 增强器未启用或模型为空，返回原图")
            return img
        
        try:
            # 拷贝图像以避免修改原图
            img = img.copy()
            print(f"[DEBUG] 输入图像形状: {img.shape}, 类型: {img.dtype}, 值范围: [{img.min()}, {img.max()}]")
            
            try:
                # 预处理图像
                print("[DEBUG] 开始预处理图像...")
                input_data = self.preprocess(img)
                print(f"[DEBUG] 预处理后数据形状: {input_data.shape}, 类型: {input_data.dtype}, 值范围: [{input_data.min()}, {input_data.max()}]")
            except Exception as e:
                print(f"[ERROR] 预处理失败: {str(e)}")
                print(f"[ERROR] 错误类型: {type(e)}")
                import traceback
                print(f"[ERROR] 堆栈跟踪:\n{traceback.format_exc()}")
                raise
            
            try:
                # 运行ONNX推理
                print("[DEBUG] 开始ONNX推理...")
                output = self.session.run(None, {self.input_name: input_data})[0]
                print(f"[DEBUG] ONNX输出形状: {output.shape}, 类型: {output.dtype}, 值范围: [{output.min()}, {output.max()}]")
            except Exception as e:
                print(f"[ERROR] ONNX推理失败: {str(e)}")
                print(f"[ERROR] 错误类型: {type(e)}")
                import traceback
                print(f"[ERROR] 堆栈跟踪:\n{traceback.format_exc()}")
                raise
            
            try:
                # 后处理输出
                print("[DEBUG] 开始后处理...")
                enhanced_img = self.postprocess(output)
                print(f"[DEBUG] 后处理后图像形状: {enhanced_img.shape}, 类型: {enhanced_img.dtype}, 值范围: [{enhanced_img.min()}, {enhanced_img.max()}]")
            except Exception as e:
                print(f"[ERROR] 后处理失败: {str(e)}")
                print(f"[ERROR] 错误类型: {type(e)}")
                import traceback
                print(f"[ERROR] 堆栈跟踪:\n{traceback.format_exc()}")
                raise
            
            try:
                # 根据增强强度混合原图和增强结果
                if self.enhancement_strength < 1.0:
                    print(f"[DEBUG] 应用增强强度: {self.enhancement_strength}")
                    # 调整原图大小以匹配增强结果
                    original_resized = cv2.resize(img, (enhanced_img.shape[1], enhanced_img.shape[0]), 
                                                interpolation=cv2.INTER_LANCZOS4)
                    enhanced_img = cv2.addWeighted(enhanced_img, self.enhancement_strength, 
                                                original_resized, 1.0 - self.enhancement_strength, 0)
            except Exception as e:
                print(f"[ERROR] 应用增强强度失败: {str(e)}")
                print(f"[ERROR] 错误类型: {type(e)}")
                import traceback
                print(f"[ERROR] 堆栈跟踪:\n{traceback.format_exc()}")
                raise
            
            try:
                # 如果启用嘴唇保护并有人脸关键点，保留原始嘴唇区域
                if self.mouth_protection and face_landmarks is not None and len(face_landmarks) > 0:
                    print("[DEBUG] 应用嘴唇保护...")
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
                            except Exception as e:
                                print(f"[ERROR] 获取嘴唇关键点失败: {str(e)}")
                                print(f"[ERROR] 错误类型: {type(e)}")
                                import traceback
                                print(f"[ERROR] 堆栈跟踪:\n{traceback.format_exc()}")
                                # 使用固定区域作为备选
                                h, w = enhanced_img.shape[:2]
                                mouth_points = np.array([
                                    [w//2 - w//8, h//2 + h//8],
                                    [w//2 + w//8, h//2 + h//8],
                                    [w//2 + w//8, h//2 + h//4],
                                    [w//2 - w//8, h//2 + h//4]
                                ])
                            
                            try:
                                # 创建嘴唇区域多边形
                                cv2.fillPoly(mask, [np.array(mouth_points, dtype=np.int32)], (255, 255, 255))
                                
                                # 扩大嘴唇区域
                                kernel = np.ones((15, 15), np.uint8)
                                mask = cv2.dilate(mask, kernel, iterations=1)
                            except Exception as e:
                                print(f"[ERROR] 创建嘴唇遮罩失败: {str(e)}")
                                print(f"[ERROR] 错误类型: {type(e)}")
                                import traceback
                                print(f"[ERROR] 堆栈跟踪:\n{traceback.format_exc()}")
                                raise
                        
                        try:
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
                            print(f"[ERROR] 应用嘴唇遮罩失败: {str(e)}")
                            print(f"[ERROR] 错误类型: {type(e)}")
                            import traceback
                            print(f"[ERROR] 堆栈跟踪:\n{traceback.format_exc()}")
                            raise
                    
                    except Exception as e:
                        print(f"[ERROR] 应用嘴唇保护时出错: {str(e)}")
                        print(f"[ERROR] 错误类型: {type(e)}")
                        import traceback
                        print(f"[ERROR] 堆栈跟踪:\n{traceback.format_exc()}")
            except Exception as e:
                print(f"[ERROR] 嘴唇保护处理失败: {str(e)}")
                print(f"[ERROR] 错误类型: {type(e)}")
                import traceback
                print(f"[ERROR] 堆栈跟踪:\n{traceback.format_exc()}")
                raise
            
            print("[DEBUG] 增强完成，返回结果")
            return enhanced_img
            
        except Exception as e:
            print(f"[ERROR] 面部增强过程出错: {str(e)}")
            print(f"[ERROR] 错误类型: {type(e)}")
            import traceback
            print(f"[ERROR] 完整错误堆栈:\n{traceback.format_exc()}")
            # 返回原图作为备选
            return img 