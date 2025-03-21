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
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        # 检查模型文件大小
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # 转换为MB
        print(f"模型文件 {model_path} 大小: {file_size:.2f} MB")
        if file_size < 0.1:  # 如果小于100KB
            raise ValueError(f"模型文件 {model_path} 大小异常: {file_size:.2f} MB")
            
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
                
            # Create ONNX session
            self.ort_session = ort.InferenceSession(model_path, providers=provider)
            
            # Print model input and output details
            for i, input_info in enumerate(self.ort_session.get_inputs()):
                print(f"模型输入 {i}: 名称={input_info.name}, 形状={input_info.shape}, 类型={input_info.type}")
            for i, output_info in enumerate(self.ort_session.get_outputs()):
                print(f"模型输出 {i}: 名称={output_info.name}, 形状={output_info.shape}, 类型={output_info.type}")
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
            if img is None:
                print("Error: 输入图像为None")
                return None
                
            print(f"预处理图像，形状: {img.shape}, 类型: {img.dtype}, 值范围: [{np.min(img)}, {np.max(img)}]")
            
            # 检查图像是否为空或包含NaN/Inf
            if img.size == 0 or np.isnan(img).any() or np.isinf(img).any():
                print(f"警告: 图像包含无效数据. 大小: {img.size}, NaN: {np.isnan(img).any()}, Inf: {np.isinf(img).any()}")
                return None
                
            h, w, c = img.shape
                
            # 转换为float32类型，并确保值范围在0-1
            if img.dtype != np.float32:
                if img.dtype == np.uint8:
                    # 如果是uint8，除以255
                    img_for_model = img.astype(np.float32) / 255.0
                else:
                    # 对于其他类型，先转换为float32
                    img_for_model = img.astype(np.float32)
                    # 如果最大值大于1.5，假设范围是0-255并归一化
                    if np.max(img_for_model) > 1.5:  
                        img_for_model /= 255.0
            else:
                img_for_model = img.copy()
                # 确保float32类型的值在0-1范围内
                if np.max(img_for_model) > 1.5:
                    img_for_model /= 255.0
                
            # 确保值在0-1范围内
            img_for_model = np.clip(img_for_model, 0, 1)
                
            # 使用标准尺寸调整图像
            if self.enhancement_method in ('gfpgan', 'codeformer'):
                target_size = 512
            elif self.enhancement_method == 'gpen':
                target_size = 512
            else:
                target_size = 512  # 默认尺寸
            
            # 保存调整大小前的图像用于调试
            try:
                debug_dir = "enhance_debug"
                os.makedirs(debug_dir, exist_ok=True)
                before_resize_path = os.path.join(debug_dir, "before_resize.png")
                before_resize = (img_for_model * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(before_resize_path, cv2.cvtColor(before_resize, cv2.COLOR_RGB2BGR))
                print(f"已保存调整大小前的图像到 {before_resize_path}")
            except Exception as e:
                print(f"保存调整大小前的图像时出错: {str(e)}")
                
            # 调整图像大小，保持宽高比
            if h != target_size or w != target_size:
                factor = target_size / max(h, w)
                new_h, new_w = int(h * factor), int(w * factor)
                img_for_model = cv2.resize(img_for_model, (new_w, new_h))
                # 添加填充以获得正方形图像
                pad_h, pad_w = target_size - new_h, target_size - new_w
                pad_top, pad_bottom = pad_h // 2, pad_h - (pad_h // 2)
                pad_left, pad_right = pad_w // 2, pad_w - (pad_w // 2)
                img_for_model = cv2.copyMakeBorder(
                    img_for_model, pad_top, pad_bottom, pad_left, pad_right, 
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
            
            # 保存调整大小后的图像用于调试
            try:
                after_resize_path = os.path.join(debug_dir, "after_resize.png")
                after_resize = (img_for_model * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(after_resize_path, cv2.cvtColor(after_resize, cv2.COLOR_RGB2BGR))
                print(f"已保存调整大小后的图像到 {after_resize_path}")
            except Exception as e:
                print(f"保存调整大小后的图像时出错: {str(e)}")
            
            # GPEN模型需要特殊处理
            if self.enhancement_method == 'gpen':
                # 针对BGR输入的模型，颜色通道需要反转
                # 从RGB转为BGR，因为GPEN模型通常以BGR格式训练
                print("检测到GPEN模型，将RGB转换为BGR")
                img_for_model = img_for_model[:, :, ::-1]
                
                # 保存GPEN特殊处理后的图像用于调试
                try:
                    gpen_input_path = os.path.join(debug_dir, "gpen_input.png")
                    gpen_input = (img_for_model * 255).clip(0, 255).astype(np.uint8)
                    cv2.imwrite(gpen_input_path, gpen_input)  # 已经是BGR，不需要转换
                    print(f"已保存GPEN输入图像到 {gpen_input_path}")
                except Exception as e:
                    print(f"保存GPEN输入图像时出错: {str(e)}")
                
            # 转换为NCHW格式
            img_for_model = img_for_model.transpose(2, 0, 1)
            img_for_model = np.expand_dims(img_for_model, axis=0)
            
            print(f"预处理后图像形状: {img_for_model.shape}, 类型: {img_for_model.dtype}, 值范围: [{np.min(img_for_model)}, {np.max(img_for_model)}]")
            
            return img_for_model
                
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
                print(f"已保存输入图像到 {input_debug_path}")
            except Exception as e:
                print(f"保存输入图像时出错: {str(e)}")
            
            # 检查输入图像是否有效
            if img is None:
                print("输入图像为None，无法增强")
                return img
                
            if img.size == 0:
                print("输入图像为空，无法增强")
                return img
                
            # 确保图像是float32类型
            if img.dtype != np.float32:
                print(f"输入图像类型为{img.dtype}，转换为float32")
                if img.dtype == np.uint8:
                    # 如果是uint8，转换为0-1范围的float32
                    img = img.astype(np.float32) / 255.0
                elif np.issubdtype(img.dtype, np.floating):
                    # 如果是其他浮点类型，确保范围合适并转换
                    if np.max(img) > 1.0:
                        # 如果值超过1，假设是0-255范围，归一化到0-1
                        img = img.astype(np.float32) / 255.0
                    else:
                        # 保持0-1范围
                        img = img.astype(np.float32)
                else:
                    # 其他类型，转换并归一化
                    img = img.astype(np.float32)
                    if np.max(img) > 1.0:
                        img = img / 255.0
                
            # 检查并处理NaN或Inf值
            if np.isnan(img).any() or np.isinf(img).any():
                print("警告：输入图像包含NaN或Inf值，将其替换为0")
                img = np.nan_to_num(img, nan=0, posinf=1.0, neginf=0)
                
            # 记录原始图像信息
            print(f"输入图像: 形状={img.shape}, 类型={img.dtype}, 非零像素={np.count_nonzero(img)}")
            print(f"图像值范围: [{np.min(img)}, {np.max(img)}]")
            
            # 保存原始图像作为备份
            original_img = img.copy()
            
            # 检查模型会话是否可用
            if not hasattr(self, 'ort_session') or self.ort_session is None:
                print("错误: ONNX会话未初始化")
                return original_img
                
            # 预处理图像
            preprocessed = self.preprocess(img, face_landmarks)
            if preprocessed is None:
                print("预处理失败，返回原始图像")
                return original_img
                
            # 保存预处理后的图像用于调试
            try:
                # 从NCHW转换回HWC用于保存
                preproc_vis = preprocessed[0].transpose(1, 2, 0)
                preproc_debug_path = os.path.join(debug_dir, "preprocessed.png")
                preproc_debug = (preproc_vis * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(preproc_debug_path, cv2.cvtColor(preproc_debug, cv2.COLOR_RGB2BGR))
                print(f"已保存预处理图像到 {preproc_debug_path}")
            except Exception as e:
                print(f"保存预处理图像时出错: {str(e)}")
            
            # 确保输入形状正确
            model_input_shape = self.ort_session.get_inputs()[0].shape
            if len(model_input_shape) == 4:  # NCHW格式
                # 修复对非数值类型的比较问题
                expected_shape = []
                for i, d in enumerate(model_input_shape):
                    # 检查d是否为数值类型
                    if isinstance(d, (int, np.integer)):
                        # 如果是数值且大于0，使用该值；否则使用预处理后图像的相应维度
                        expected_shape.append(d if d > 0 else preprocessed.shape[i])
                    else:
                        # 如果不是数值类型(比如字符串或None)，使用预处理后图像的相应维度
                        print(f"警告: 模型形状维度 {i} 为非数值类型 ({type(d)}): {d}")
                        expected_shape.append(preprocessed.shape[i])
                
                expected_shape = tuple(expected_shape)
                
                if preprocessed.shape != expected_shape:
                    print(f"警告: 输入形状不匹配. 预期 {expected_shape}, 实际 {preprocessed.shape}")
                    # 尝试调整形状以匹配模型预期
                    try:
                        if np.prod(preprocessed.shape) == np.prod(expected_shape):
                            preprocessed = preprocessed.reshape(expected_shape)
                        else:
                            print("形状不兼容，无法调整")
                            return original_img
                    except Exception as reshape_error:
                        print(f"调整形状时出错: {reshape_error}")
                        return original_img
                
            # 进行模型推理
            print("开始ONNX推理")
            try:
                # 获取模型的输入名称
                input_name = self.ort_session.get_inputs()[0].name
                print(f"使用输入名称: {input_name}")
                
                # 确保所有输入名称有效
                input_feed = {input_name: preprocessed}
                
                # 如果是GPEN模型，检查是否有额外输入参数
                if self.enhancement_method == 'gpen':
                    print("准备GPEN模型的输入")
                    for i, input_info in enumerate(self.ort_session.get_inputs()):
                        if i > 0:  # 第一个输入已经处理过
                            print(f"额外输入 {i}: {input_info.name}, 形状: {input_info.shape}")
                            
                            # 处理可能的额外输入
                            if "return_rgb" in input_info.name.lower():
                                # GPEN可能有一个控制输出颜色空间的布尔输入
                                input_feed[input_info.name] = np.array([True], dtype=np.bool_)
                                print(f"设置 {input_info.name} = True")
                
                # 执行推理
                outputs = self.ort_session.run(None, input_feed)
                print(f"ONNX推理完成，输出长度: {len(outputs)}")
                
                # 检查输出是否有效
                if len(outputs) == 0:
                    print("模型未产生输出，返回原始图像")
                    return original_img
                    
                output = outputs[0]
                print(f"模型输出: 形状={output.shape}, 类型={output.dtype}, 值范围=[{np.min(output)}, {np.max(output)}]")
                print(f"输出非零元素数量: {np.count_nonzero(output)}/{output.size} ({np.count_nonzero(output)/output.size*100:.2f}%)")
                
                # 保存原始模型输出用于调试
                try:
                    # 从NCHW转换回HWC用于保存
                    if len(output.shape) == 4:  # NCHW格式
                        raw_output_vis = output[0].transpose(1, 2, 0)
                    else:
                        raw_output_vis = output
                    
                    raw_output_debug_path = os.path.join(debug_dir, "raw_model_output.png")
                    # 确保值在有效范围内
                    if np.max(raw_output_vis) <= 1.0:
                        raw_output_debug = (raw_output_vis * 255).clip(0, 255).astype(np.uint8)
                    else:
                        raw_output_debug = raw_output_vis.clip(0, 255).astype(np.uint8)
                    
                    cv2.imwrite(raw_output_debug_path, cv2.cvtColor(raw_output_debug, cv2.COLOR_RGB2BGR))
                    print(f"已保存原始模型输出到 {raw_output_debug_path}")
                except Exception as e:
                    print(f"保存模型输出时出错: {str(e)}")
                    traceback.print_exc()
                
                # 检查NaN和Inf值
                if np.isnan(output).any() or np.isinf(output).any():
                    print(f"警告: 模型输出包含NaN或Inf值. NaN: {np.isnan(output).any()}, Inf: {np.isinf(output).any()}")
                    output = np.nan_to_num(output, nan=0, posinf=1, neginf=0)
                
                # 检查输出是否为空或全零
                if output.size == 0 or np.count_nonzero(output) == 0:
                    print("模型输出为空或全零，返回原始图像")
                    return original_img
                
            except Exception as e:
                print(f"ONNX推理时出错: {str(e)}")
                traceback.print_exc()
                return original_img
                
            # 后处理
            try:
                result = self.postprocess(img, output, face_landmarks)
                if result is None:
                    print("后处理返回None，使用原始图像")
                    return original_img
                    
                # 保存最终结果用于调试
                try:
                    result_debug_path = os.path.join(debug_dir, "final_result.png")
                    result_debug = (result * 255).clip(0, 255).astype(np.uint8) if np.max(result) <= 1.0 else result.clip(0, 255).astype(np.uint8)
                    cv2.imwrite(result_debug_path, cv2.cvtColor(result_debug, cv2.COLOR_RGB2BGR))
                    print(f"已保存最终结果到 {result_debug_path}")
                except Exception as e:
                    print(f"保存最终结果时出错: {str(e)}")
                    
                print(f"最终结果: 形状={result.shape}, 类型={result.dtype}, 值范围=[{np.min(result)}, {np.max(result)}]")
                print(f"最终结果非零元素数量: {np.count_nonzero(result)}/{result.size} ({np.count_nonzero(result)/result.size*100:.2f}%)")
                return result
            except Exception as e:
                print(f"后处理时出错: {str(e)}")
                traceback.print_exc()
                return original_img
            
        except Exception as e:
            print(f"增强过程中发生未捕获的错误: {str(e)}")
            traceback.print_exc()
            # 如果有原始图像则返回，否则返回输入图像
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
        try:
            # 创建调试目录
            debug_dir = "enhance_debug"
            os.makedirs(debug_dir, exist_ok=True)
            
            print(f"后处理开始：输入图像形状={img.shape}, 输出形状={output.shape}")
            
            # 确保输出格式兼容
            if output.dtype != np.float32:
                print(f"正在将输出从 {output.dtype} 转换为 float32")
                # 特别处理float16，OpenCV不支持
                if output.dtype.name == 'float16':
                    print("检测到float16输出，转换为float32")
                output = output.astype(np.float32)
                
            # 从NCHW转换为HWC格式
            if len(output.shape) == 4:  # NCHW格式
                output = output[0].transpose(1, 2, 0)
                
            # 保存转换后的输出用于调试
            try:
                transposed_debug_path = os.path.join(debug_dir, "transposed_output.png")
                # 确保值在有效范围内
                if np.max(output) <= 1.0:
                    transposed_debug = (output * 255).clip(0, 255).astype(np.uint8)
                else:
                    transposed_debug = output.clip(0, 255).astype(np.uint8)
                
                cv2.imwrite(transposed_debug_path, cv2.cvtColor(transposed_debug, cv2.COLOR_RGB2BGR))
                print(f"已保存转置后输出到 {transposed_debug_path}")
            except Exception as e:
                print(f"保存转置后输出时出错: {str(e)}")
                traceback.print_exc()
            
            h, w, c = img.shape
            out_h, out_w, out_c = output.shape
            
            print(f"输入尺寸: {w}x{h}, 输出尺寸: {out_w}x{out_h}")
            
            # GPEN模型需要特殊处理
            if self.enhancement_method == 'gpen':
                print("检测到GPEN模型输出，将BGR转换为RGB")
                # 从BGR转回RGB
                output = output[:, :, ::-1]
                
                # 保存颜色转换后的GPEN输出
                try:
                    gpen_output_path = os.path.join(debug_dir, "gpen_output_rgb.png")
                    if np.max(output) <= 1.0:
                        gpen_output = (output * 255).clip(0, 255).astype(np.uint8)
                    else:
                        gpen_output = output.clip(0, 255).astype(np.uint8)
                    cv2.imwrite(gpen_output_path, cv2.cvtColor(gpen_output, cv2.COLOR_RGB2BGR))
                    print(f"已保存GPEN RGB输出到 {gpen_output_path}")
                except Exception as e:
                    print(f"保存GPEN RGB输出时出错: {str(e)}")
                    traceback.print_exc()
            
            # 重新调整大小以匹配输入图像
            if out_h != h or out_w != w:
                # 确保使用float32格式
                output = cv2.resize(output, (w, h))
                
            # 确保值范围为0-1
            output_max = np.max(output)
            if output_max > 1.5:
                # 如果最大值大于1.5，假设范围是0-255，归一化到0-1
                output = output / 255.0
                
            # 确保限制在0-1范围内
            output = np.clip(output, 0, 1)
            
            # 保存裁剪后的输出用于调试
            try:
                clipped_debug_path = os.path.join(debug_dir, "clipped_output.png")
                clipped_debug = (output * 255).astype(np.uint8)
                cv2.imwrite(clipped_debug_path, cv2.cvtColor(clipped_debug, cv2.COLOR_RGB2BGR))
                print(f"已保存裁剪后输出到 {clipped_debug_path}")
            except Exception as e:
                print(f"保存裁剪后输出时出错: {str(e)}")
                traceback.print_exc()
                
            # 确保输出图像为float32
            output = output.astype(np.float32)
                
            # 应用增强强度
            result = img * (1 - self.enhancement_strength) + output * self.enhancement_strength
            
            # 确保结果为float32且在0-1范围内
            result = np.clip(result, 0, 1).astype(np.float32)
            
            # 保存混合后的结果用于调试
            try:
                blended_debug_path = os.path.join(debug_dir, "blended_result.png")
                blended_debug = (result * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(blended_debug_path, cv2.cvtColor(blended_debug, cv2.COLOR_RGB2BGR))
                print(f"已保存混合后结果到 {blended_debug_path}")
            except Exception as e:
                print(f"保存混合后结果时出错: {str(e)}")
                traceback.print_exc()
            
            # 应用嘴部保护
            if self.mouth_protection and face_landmarks is not None:
                try:
                    # 创建嘴部遮罩
                    mouth_mask = np.zeros((h, w), dtype=np.float32)
                    if len(face_landmarks) >= 68:  # 完整的68点面部关键点
                        # 对于68点关键点，嘴部通常是索引48-67
                        mouth_points = face_landmarks[48:68]
                    elif len(face_landmarks) == 5:  # 5点关键点(眼睛*2, 鼻子, 嘴角*2)
                        # 对于5点关键点，嘴部通常是最后2个点
                        mouth_points = face_landmarks[3:5]
                    else:
                        # 如果关键点数量不是预期的，尝试计算面部下半部分
                        mouth_points = face_landmarks[len(face_landmarks)//2:]
                        
                    # 转换关键点为轮廓格式
                    mouth_contour = np.array(mouth_points, dtype=np.int32)
                    
                    # 在遮罩上绘制嘴部区域并填充
                    cv2.fillPoly(mouth_mask, [mouth_contour], 1.0)
                    
                    # 应用高斯模糊使过渡更平滑
                    mouth_mask = cv2.GaussianBlur(mouth_mask, (51, 51), 15)
                    
                    # 应用嘴部保护强度
                    mouth_mask = mouth_mask * self.mouth_protection_strength
                    
                    # 扩展遮罩为3通道
                    mouth_mask_3d = np.repeat(mouth_mask[:, :, np.newaxis], 3, axis=2)
                    
                    # 使用遮罩混合原始图像和增强图像
                    # 嘴部保护: 在嘴部区域更多地使用原始图像
                    result = img * mouth_mask_3d + result * (1 - mouth_mask_3d)
                    
                except Exception as e:
                    print(f"应用嘴部保护时出错: {str(e)}")
                    traceback.print_exc()
                    # 继续处理，无需中断
            
            print(f"后处理完成，结果形状={result.shape}, 类型={result.dtype}")
            return result
            
        except Exception as e:
            print(f"后处理时发生错误: {str(e)}")
            traceback.print_exc()
            return img  # 返回原始图像作为备份