import os
import cv2
import numpy as np
from typing import Union, Optional, Tuple, List
import warnings
import onnxruntime as ort
import time
import traceback

class FaceEnhancer:
    """人脸增强器，支持使用ONNX模型的GPEN，GFPGAN和CodeFormer三种增强方式"""
    
    def __init__(self, enhancement_method='gfpgan', enhancement_strength=0.5, model_path=None, 
                 mouth_protection=True, mouth_protection_strength=0.5):
        """
        Initialize a face enhancer.
        Args:
            enhancement_method: 'gfpgan', 'gpen' or 'codeformer'
            enhancement_strength: how much enhancement to apply, 0 to 1
            model_path: path to the model
            mouth_protection: whether to protect the mouth area when enhancing
            mouth_protection_strength: how much mouth protection to apply, 0 to 1
        """
        self.enhancement_method = enhancement_method.lower()
        self.enhancement_strength = enhancement_strength
        self.mouth_protection = mouth_protection
        self.mouth_protection_strength = mouth_protection_strength
        
        # 检查模型路径是否有效
        if model_path is None:
            raise ValueError(f"必须提供模型路径!")
        
        # 详细记录模型文件位置
        print(f"加载模型, 方法: {enhancement_method}, 文件路径: {model_path}")
        print(f"当前工作目录: {os.getcwd()}")
        
        # 检查模型文件绝对路径
        model_abs_path = os.path.abspath(model_path)
        print(f"模型绝对路径: {model_abs_path}")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}, 绝对路径: {model_abs_path}")
            
        # 检查模型文件是否可读
        if not os.access(model_path, os.R_OK):
            raise PermissionError(f"模型文件存在但无法读取: {model_path}")
            
        # 检查模型文件大小
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # 转换为MB
        print(f"模型文件 {model_path} 大小: {file_size:.2f} MB")
        if file_size < 1:  # 通常ONNX模型至少几MB
            print(f"警告: 模型文件 {model_path} 大小异常小: {file_size:.2f} MB，可能不是有效的ONNX模型")
            
        # 检查文件类型
        try:
            with open(model_path, 'rb') as f:
                header = f.read(6)
                if header[:4] != b'ONNX':  # ONNX模型文件通常以"ONNX"开头
                    print(f"警告: 文件头部不是标准ONNX格式: {header}")
        except Exception as e:
            print(f"检查模型文件头部时出错: {str(e)}")
            
        print(f"Face Enhancer: 使用 {enhancement_method} 模型，路径 {model_path}")
        print(f"增强强度: {enhancement_strength}, 嘴部保护: {mouth_protection}, 嘴部保护强度: {mouth_protection_strength}")
        
        try:
            # Get available providers
            available_providers = ort.get_available_providers()
            print(f"可用的ONNX提供程序: {available_providers}")
            
            # Select provider based on availability
            if 'CUDAExecutionProvider' in available_providers:
                provider = ['CUDAExecutionProvider']
                print("使用CUDA执行提供程序")
            else:
                provider = ['CPUExecutionProvider']
                print("使用CPU执行提供程序")
                
            # 详细记录ONNX运行时版本
            print(f"ONNX运行时版本: {ort.__version__}")
            
            # 设置ONNX会话选项
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Create ONNX session with more detailed error handling
            print(f"正在创建ONNX会话: {model_path}")
            self.ort_session = ort.InferenceSession(model_path, sess_options=session_options, providers=provider)
            print("ONNX会话创建成功")
            
            # 获取模型输入分辨率
            if self.enhancement_method == 'gpen':
                self.resolution = self.ort_session.get_inputs()[0].shape[-2:]
                if not all(self.resolution):  # 如果有任意一个维度是0或None
                    self.resolution = (512, 512)  # 默认分辨率
                print(f"GPEN模型输入分辨率: {self.resolution}")
                
                # 获取GPEN模型输入名称
                self.input_name = self.ort_session.get_inputs()[0].name
                print(f"GPEN模型输入名称: {self.input_name}")
            
            # Print model input and output details
            print("===== 模型输入节点信息 =====")
            for i, input_info in enumerate(self.ort_session.get_inputs()):
                print(f"模型输入 {i}: 名称={input_info.name}, 形状={input_info.shape}, 类型={input_info.type}")
                
            print("===== 模型输出节点信息 =====")
            for i, output_info in enumerate(self.ort_session.get_outputs()):
                print(f"模型输出 {i}: 名称={output_info.name}, 形状={output_info.shape}, 类型={output_info.type}")
                
            # 模型特定信息处理
            if self.enhancement_method == 'gpen':
                print("===== GPEN模型特殊信息 =====")
                # 推测模型类型
                model_name = os.path.basename(model_path)
                print(f"模型文件名: {model_name}")
                
                # 根据不同GPEN模型设置特定参数
                if "BFR" in model_name:
                    print("检测到BFR变体的GPEN模型")
                    self.gpen_variant = "BFR"
                elif "Blind" in model_name:
                    print("检测到Blind变体的GPEN模型")
                    self.gpen_variant = "Blind"
                else:
                    print("未能确定GPEN变体类型，使用默认设置")
                    self.gpen_variant = "default"
                
        except Exception as e:
            print(f"创建ONNX会话时发生错误: {str(e)}")
            print(f"尝试加载模型: {model_path}")
            print(f"当前工作目录: {os.getcwd()}")
            traceback.print_exc()
            raise RuntimeError(f"无法初始化面部增强器: {str(e)}")
    
    def preprocess(self, img, face_landmarks=None):
        """
        Preprocess image for the model.
        Args:
            img: input image, RGB order
            face_landmarks: face landmarks if available

        Returns:
            Preprocessed image
        """
        try:
            # 创建调试目录
            debug_dir = "enhance_debug"
            os.makedirs(debug_dir, exist_ok=True)
            
            # 保存原始输入
            try:
                raw_input_path = os.path.join(debug_dir, "raw_input.png")
                raw_input = img.astype(np.uint8) if np.max(img) <= 1.0 else img.clip(0, 255).astype(np.uint8)
                cv2.imwrite(raw_input_path, cv2.cvtColor(raw_input, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"保存原始输入出错: {str(e)}")
            
            # 检查输入是否有效
            if img is None or img.size == 0:
                print("错误: 输入图像为None或空")
                return None
                
            if np.isnan(img).any() or np.isinf(img).any():
                print("警告: 图像包含无效数据")
                img = np.nan_to_num(img, nan=0, posinf=1, neginf=0)
                
            # GPEN预处理 (更简洁的实现)
            if self.enhancement_method == 'gpen':
                # 确保图像为uint8类型
                if np.max(img) <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
                
                # 调整大小到模型分辨率
                img_resized = cv2.resize(img, self.resolution, interpolation=cv2.INTER_LINEAR)
                
                # RGB转BGR并归一化到[0,1]
                img_bgr = img_resized[:,:,::-1].astype(np.float32) / 255.0
                
                # 保存预处理后图像用于调试
                try:
                    preproc_path = os.path.join(debug_dir, "preprocessed.png")
                    preproc_vis = (img_bgr[:,:,::-1] * 255).astype(np.uint8)  # 转回RGB用于显示
                    cv2.imwrite(preproc_path, cv2.cvtColor(preproc_vis, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    print(f"保存预处理图像出错: {str(e)}")
                
                # 转换为CHW格式
                img_chw = img_bgr.transpose((2, 0, 1))
                
                # 归一化到[-1,1]范围
                img_norm = (img_chw - 0.5) / 0.5
                
                # 添加批次维度
                img_batch = np.expand_dims(img_norm, axis=0).astype(np.float32)
                
                # 保存GPEN输入
                try:
                    gpen_input_path = os.path.join(debug_dir, "gpen_input.png")
                    gpen_input_vis = ((img_norm.transpose(1, 2, 0) + 1) / 2)
                    gpen_input_vis = (gpen_input_vis[:,:,::-1] * 255).astype(np.uint8)  # BGR转RGB并缩放到255
                    cv2.imwrite(gpen_input_path, gpen_input_vis)
                except Exception as e:
                    print(f"保存GPEN输入出错: {str(e)}")
                
                print(f"GPEN预处理完成，输入形状: {img_batch.shape}")
                return img_batch
                
            # 其他方法的预处理保持不变...
            # ...此处省略其他方法的预处理代码...
                
        except Exception as e:
            print(f"预处理图像时出错: {str(e)}")
            traceback.print_exc()
            return None
    
    def enhance(self, img, face_landmarks=None):
        """
        Enhance a face image.
        Args:
            img: input image, RGB order
            face_landmarks: face landmarks if available

        Returns:
            Enhanced image
        """
        try:
            # 创建调试目录
            debug_dir = "enhance_debug"
            os.makedirs(debug_dir, exist_ok=True)
            
            # 保存输入图像用于调试
            try:
                input_debug_path = os.path.join(debug_dir, "input_image.png")
                input_debug = (img * 255).clip(0, 255).astype(np.uint8) if np.max(img) <= 1.0 else img.clip(0, 255).astype(np.uint8)
                cv2.imwrite(input_debug_path, cv2.cvtColor(input_debug, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"保存输入图像时出错: {str(e)}")
            
            # 检查输入图像是否有效
            if img is None or img.size == 0:
                print("输入图像无效，无法增强")
                return img
                
            # 记录原始图像
            original_img = img.copy()
            print(f"输入图像: 形状={img.shape}, 类型={img.dtype}")
            
            # 检查模型会话是否可用
            if not hasattr(self, 'ort_session') or self.ort_session is None:
                print("错误: ONNX会话未初始化")
                return original_img
                
            # 保存原始图像尺寸
            orig_h, orig_w = img.shape[:2]
            
            # 预处理图像
            preprocessed = self.preprocess(img, face_landmarks)
            if preprocessed is None:
                print("预处理失败，返回原始图像")
                return original_img
                
            # 执行推理 (GPEN简化实现)
            if self.enhancement_method == 'gpen':
                try:
                    print("执行GPEN推理...")
                    # 使用模型本身的输入名称进行推理而不是固定的'input'
                    input_feed = {self.input_name: preprocessed}
                    print(f"使用输入名称 '{self.input_name}' 进行推理")
                    
                    outputs = self.ort_session.run(None, input_feed)
                    
                    if not outputs or len(outputs) == 0:
                        print("GPEN模型未产生输出")
                        return original_img
                        
                    # 获取第一个输出
                    output = outputs[0][0]
                    print(f"GPEN输出: 形状={output.shape}, 类型={output.dtype}")
                    
                    # 后处理
                    # 从[-1,1]转回[0,1]
                    output_norm = (output.transpose(1, 2, 0).clip(-1, 1) + 1) / 2
                    # BGR转RGB并缩放到255
                    output_rgb = (output_norm[:,:,::-1] * 255).clip(0, 255).astype(np.uint8)
                    
                    # 保存原始输出
                    try:
                        raw_output_path = os.path.join(debug_dir, "raw_model_output.png")
                        cv2.imwrite(raw_output_path, cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))
                    except Exception as e:
                        print(f"保存原始输出出错: {str(e)}")
                    
                    # 调整大小以匹配输入
                    if (output_rgb.shape[0] != orig_h or output_rgb.shape[1] != orig_w):
                        output_rgb = cv2.resize(output_rgb, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    
                    # 保存调整大小后的输出
                    try:
                        resized_output_path = os.path.join(debug_dir, "gpen_output_rgb.png")
                        cv2.imwrite(resized_output_path, cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))
                    except Exception as e:
                        print(f"保存调整大小后输出出错: {str(e)}")
                    
                    # 确保原图和增强图像类型一致
                    original_for_blend = original_img
                    if np.max(original_img) <= 1.0:
                        # 如果原图是[0,1]范围，将输出转换为相同范围
                        output_rgb = output_rgb.astype(np.float32) / 255.0
                    else:
                        # 如果原图是[0,255]范围，将原图转换为uint8
                        if original_img.dtype != np.uint8:
                            original_for_blend = original_img.clip(0, 255).astype(np.uint8)
                    
                    # 混合原图与增强结果
                    result = cv2.addWeighted(
                        output_rgb, self.enhancement_strength,
                        original_for_blend, 1.0 - self.enhancement_strength,
                        0
                    )
                    
                    # 保存最终结果
                    try:
                        result_path = os.path.join(debug_dir, "final_result.png")
                        result_vis = result
                        if np.max(result) <= 1.0:
                            result_vis = (result * 255).clip(0, 255).astype(np.uint8)
                        cv2.imwrite(result_path, cv2.cvtColor(result_vis, cv2.COLOR_RGB2BGR))
                    except Exception as e:
                        print(f"保存最终结果出错: {str(e)}")
                    
                    # 应用嘴部保护如果需要
                    if self.mouth_protection and face_landmarks is not None:
                        # ...此处省略嘴部保护代码...
                        pass
                    
                    print("GPEN增强完成")
                    return result
                    
                except Exception as e:
                    print(f"GPEN推理时出错: {str(e)}")
                    traceback.print_exc()
                    return original_img
            
            # 其他增强方法保持不变...
            # ...此处省略其他增强方法代码...
            
            return original_img
            
        except Exception as e:
            print(f"增强过程中发生未捕获的错误: {str(e)}")
            traceback.print_exc()
            return original_img if 'original_img' in locals() else img
    
    def postprocess(self, img, output, face_landmarks=None):
        """
        Postprocess the output of the model.
        Args:
            img: input image, RGB order
            output: output of the model
            face_landmarks: face landmarks if available
            
        Returns:
            Postprocessed image
        """
        # GPEN使用enhance方法中集成的后处理，此方法用于其他增强器
        print("GPEN不使用单独的postprocess方法，已在enhance方法中处理")
        return img