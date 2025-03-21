import os
import cv2
import numpy as np
from typing import Union, Optional, Tuple, List
import warnings
import onnxruntime as ort
import time
import traceback
import torch

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
                self.device = 'cuda'
            else:
                provider = ['CPUExecutionProvider']
                print("使用CPU执行提供程序")
                self.device = 'cpu'
                
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
    
    def preprocess(self, img):
        """
        Preprocess the input image for the model.
        
        Args:
            img: Input image in BGR format (OpenCV) or RGB format (PIL)
            
        Returns:
            Preprocessed image ready for the model
        """
        debug_dir = os.path.join(os.getcwd(), "debug_images")
        os.makedirs(debug_dir, exist_ok=True)
        
        # 保存原始输入图像
        raw_input_path = os.path.join(debug_dir, "raw_input.png")
        if img is None:
            print(f"[Error] Input image is None")
            return None
            
        # 检查输入图像是否有有效数据
        if not img.size or np.isnan(img).any() or np.isinf(img).any():
            print(f"[Error] Input image contains invalid data: size={img.size if hasattr(img, 'size') else 'N/A'}, has NaN={np.isnan(img).any() if isinstance(img, np.ndarray) else 'N/A'}, has Inf={np.isinf(img).any() if isinstance(img, np.ndarray) else 'N/A'}")
            return None
            
        # 保存原始输入图像
        try:
            if np.max(img) <= 1.0:
                input_debug = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                input_debug = img.clip(0, 255).astype(np.uint8)
            # 输入已经是BGR格式，直接保存
            cv2.imwrite(raw_input_path, input_debug)
            print(f"[Debug] Saved raw input image, shape={img.shape}, dtype={img.dtype}, range=[{np.min(img)}, {np.max(img)}]")
        except Exception as e:
            print(f"[Error] Failed to save raw input image: {e}")
        
        # 保存调整大小前的图像
        before_resize_path = os.path.join(debug_dir, "before_resize.png")
        try:
            before_resize_img = input_debug.copy()  # 使用已转换的input_debug
            cv2.imwrite(before_resize_path, before_resize_img)
            print(f"[Debug] Saved image before resize, shape={before_resize_img.shape}")
        except Exception as e:
            print(f"[Error] Failed to save image before resize: {e}")
        
        # 确保输入图像是np.uint8类型，并保持BGR格式
        if img.dtype != np.uint8:
            if np.max(img) <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)
            print(f"[Debug] Converted input to uint8, shape={img.shape}, dtype={img.dtype}, range=[{np.min(img)}, {np.max(img)}]")
        
        # 保持原始图像纵横比的同时调整大小
        h, w = img.shape[:2]
        size = self.resolution[0]  # Assuming self.resolution is a tuple
        
        # 计算新的高度和宽度，保持纵横比
        if h > w:
            new_h = size
            new_w = int(w * size / h)
        else:
            new_w = size
            new_h = int(h * size / w)
        
        # 调整图像大小，保持纵横比
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 创建一个空白图像，形状为预期的方形
        square_img = np.zeros((size, size, 3), dtype=np.uint8)
        
        # 计算居中的位置
        y_offset = (size - new_h) // 2
        x_offset = (size - new_w) // 2
        
        # 将调整大小的图像放在方形画布的中央
        square_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
        
        # 保存调整大小后的图像
        after_resize_path = os.path.join(debug_dir, "after_resize.png")
        try:
            cv2.imwrite(after_resize_path, square_img)
            print(f"[Debug] Saved image after resize, shape={square_img.shape}")
        except Exception as e:
            print(f"[Error] Failed to save image after resize: {e}")
        
        # square_img 已经是BGR格式，GPEN模型需要BGR格式输入
        # 保存最终输入到GPEN模型的图像
        gpen_input_path = os.path.join(debug_dir, "gpen_input.png")
        try:
            cv2.imwrite(gpen_input_path, square_img)
            print(f"[Debug] Saved GPEN input image, shape={square_img.shape}")
        except Exception as e:
            print(f"[Error] Failed to save GPEN input image: {e}")
        
        # 转换为模型需要的格式
        img = square_img.astype('float32') / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = torch.from_numpy(img).unsqueeze(0)  # CHW -> NCHW
        print(f"[Debug] Preprocessed tensor shape: {img.shape}, type: {img.dtype}")
        
        return img
    
    def enhance(self, img):
        """
        Enhance a face image.
        Args:
            img: Input image in RGB format (HWC)
            
        Returns:
            Enhanced image in RGB format (HWC)
        """
        print(f"[Debug] Input to enhance: shape={img.shape if img is not None else 'None'}, dtype={img.dtype if img is not None else 'None'}")
        
        # 保存输入图像
        debug_dir = os.path.join(os.getcwd(), "debug_images")
        os.makedirs(debug_dir, exist_ok=True)
        
        # 保存原始增强器输入
        enhancer_input_path = os.path.join(debug_dir, "enhancer_input_rgb.png")
        if np.max(img) <= 1.0:
            input_debug = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            input_debug = img.clip(0, 255).astype(np.uint8)
        cv2.imwrite(enhancer_input_path, cv2.cvtColor(input_debug, cv2.COLOR_RGB2BGR))
        
        # 转换输入图像为BGR格式，因为GPEN模型需要BGR输入
        if np.max(img) <= 1.0:
            img_bgr = cv2.cvtColor((img * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(img.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # 保存转换后的BGR图像
        bgr_input_path = os.path.join(debug_dir, "enhancer_input_bgr.png")
        cv2.imwrite(bgr_input_path, img_bgr)
        
        # 预处理输入图像
        inp = self.preprocess(img_bgr)
        if inp is None:
            print("[Error] Failed to preprocess input image")
            return img  # 返回原始图像
        
        # 运行ONNX模型推理
        try:
            # 为ONNX输入准备numpy数组
            if isinstance(inp, torch.Tensor):
                inp_numpy = inp.cpu().numpy()
            else:
                inp_numpy = inp
                
            # 确保输入是正确的格式
            print(f"[Debug] Model input: shape={inp_numpy.shape}, type={inp_numpy.dtype}")
            
            # 运行ONNX推理
            feed = {self.input_name: inp_numpy}
            output = self.ort_session.run(None, feed)
            
            # 保存原始模型输出以进行调试
            raw_output_path = os.path.join(debug_dir, "raw_model_output.png")
            try:
                # 使用输出数组的第一个元素（模型可能有多个输出）
                output_arr = output[0]
                
                print(f"[Debug] Raw model output: shape={output_arr.shape}, type={type(output_arr)}, min={np.min(output_arr)}, max={np.max(output_arr)}")
                
                if len(output_arr.shape) == 4:  # NCHW format
                    output_arr = output_arr[0]  # Remove batch dimension
                
                # Transpose if needed (CHW -> HWC)
                if output_arr.shape[0] == 3 and len(output_arr.shape) == 3:
                    output_arr = output_arr.transpose(1, 2, 0)  # CHW -> HWC
                
                # 确保输出有3个颜色通道
                if output_arr.shape[2] != 3:
                    print(f"[Error] Output has incorrect number of channels: {output_arr.shape}")
                    return img
                
                # 保存原始输出 (BGR格式)
                output_debug = (output_arr * 255.0).clip(0, 255).astype(np.uint8) if np.max(output_arr) <= 1.0 else output_arr.clip(0, 255).astype(np.uint8)
                cv2.imwrite(raw_output_path, output_debug)
                print(f"[Debug] Saved raw model output (BGR), shape={output_debug.shape}")
                
                # GPEN输出是BGR格式，转换为RGB以返回
                output_rgb = cv2.cvtColor(output_debug, cv2.COLOR_BGR2RGB)
                transposed_output_path = os.path.join(debug_dir, "gpen_output_rgb.png")
                cv2.imwrite(transposed_output_path, cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))  # 为了保存需要转回BGR
                print(f"[Debug] Saved GPEN output (RGB), shape={output_rgb.shape}")
                
                # 将输出归一化到0-1范围，与输入保持一致
                if np.max(img) <= 1.0:
                    output_rgb = output_rgb.astype(np.float32) / 255.0
                
                return output_rgb
            except Exception as e:
                print(f"[Error] Failed to process model output: {e}")
                traceback.print_exc()
                return img
        except Exception as e:
            print(f"[Error] ONNX inference failed: {e}")
            traceback.print_exc()
            return img
    
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