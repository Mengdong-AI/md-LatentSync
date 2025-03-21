import os
import cv2
import torch
import numpy as np
from typing import Union, Optional, Tuple, List
import warnings

class FaceEnhancer:
    """面部增强器，支持GPEN、GFPGAN和CodeFormer三种增强方法"""
    
    def __init__(
        self, 
        method: str = 'gfpgan',
        enhancement_strength: float = 0.8,
        upscale: int = 1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        mouth_protection: bool = True,  # 默认启用嘴唇保护
        mouth_protection_strength: float = 0.8,  # 嘴唇保护强度，0表示完全保留原始嘴唇，1表示完全使用增强嘴唇
        model_path: Optional[str] = None
    ):
        """
        初始化面部增强器
        
        参数:
            method (str): 增强方法，可选 'gpen', 'gfpgan', 'codeformer'
            enhancement_strength (float): 增强强度，取值范围 [0, 1]
            upscale (int): 上采样倍数，通常为1或2
            device (str): 设备，'cuda'或'cpu'
            mouth_protection (bool): 是否保护嘴唇区域，减少对唇形同步的影响
            mouth_protection_strength (float): 嘴唇保护强度，取值范围 [0, 1]，0表示完全保留原始嘴唇
            model_path (str, optional): 模型路径，如果不指定则使用默认路径
        """
        self.method = method.lower()
        self.enhancement_strength = enhancement_strength
        self.upscale = upscale
        self.device = device
        self.mouth_protection = mouth_protection
        self.mouth_protection_strength = max(0.0, min(1.0, mouth_protection_strength))
        self.model = None
        self.model_path = model_path
        
        print(f"正在初始化面部增强器 - 方法: {method}, 强度: {enhancement_strength}")
        print(f"嘴唇保护: {'已启用' if mouth_protection else '已禁用'}, 保护强度: {mouth_protection_strength}")
        
        # 确保当前目录存在模型文件夹
        os.makedirs('models/faceenhancer', exist_ok=True)
        
        # 根据方法加载相应模型
        if self.method == 'gfpgan':
            self._load_gfpgan()
        elif self.method == 'codeformer':
            self._load_codeformer()
        elif self.method == 'gpen':
            self._load_gpen()
        else:
            raise ValueError(f"不支持的增强方法: {method}，请选择 'gpen', 'gfpgan' 或 'codeformer'")
    
    def _load_gfpgan(self):
        """加载GFPGAN模型"""
        try:
            from gfpgan import GFPGANer
            
            # 设定模型路径
            if self.model_path is None:
                self.model_path = 'models/faceenhancer/GFPGANv1.4.pth'
                
                # 如果模型不存在，打印下载说明
                if not os.path.exists(self.model_path):
                    print(f"GFPGAN模型不存在: {self.model_path}")
                    print("请从 https://github.com/TencentARC/GFPGAN/releases 下载GFPGANv1.4.pth")
                    print(f"并将其放置到 {self.model_path}")
                    raise FileNotFoundError(f"GFPGAN模型不存在: {self.model_path}")
            
            # 初始化GFPGAN
            self.model = GFPGANer(
                model_path=self.model_path,
                upscale=self.upscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device
            )
            print("GFPGAN模型加载成功")
            
        except ImportError:
            print("无法导入GFPGAN，请安装相应的依赖: pip install gfpgan")
            raise
    
    def _load_codeformer(self):
        """加载CodeFormer模型"""
        try:
            from codeformer.app import CodeFormerRestorer
            
            # 设定模型路径
            if self.model_path is None:
                self.model_path = 'models/faceenhancer/codeformer.pth'
                
                # 如果模型不存在，打印下载说明
                if not os.path.exists(self.model_path):
                    print(f"CodeFormer模型不存在: {self.model_path}")
                    print("请从 https://github.com/sczhou/CodeFormer/releases 下载模型")
                    print(f"并将其放置到 {self.model_path}")
                    raise FileNotFoundError(f"CodeFormer模型不存在: {self.model_path}")
            
            # 初始化CodeFormer
            self.model = CodeFormerRestorer(
                model_path=self.model_path,
                upscale=self.upscale,
                device=self.device
            )
            print("CodeFormer模型加载成功")
            
        except ImportError:
            print("无法导入CodeFormer，请安装相应的依赖")
            raise
    
    def _load_gpen(self):
        """加载GPEN模型"""
        try:
            # 这里简化了GPEN的导入和初始化，实际使用时需要根据GPEN库的API进行调整
            from gpen.gpen_model import GPEN
            
            # 设定模型路径
            if self.model_path is None:
                self.model_path = 'models/faceenhancer/GPEN-BFR-512.pth'
                
                # 如果模型不存在，打印下载说明
                if not os.path.exists(self.model_path):
                    print(f"GPEN模型不存在: {self.model_path}")
                    print("请从 https://github.com/yangxy/GPEN/releases 下载模型")
                    print(f"并将其放置到 {self.model_path}")
                    raise FileNotFoundError(f"GPEN模型不存在: {self.model_path}")
            
            # 初始化GPEN
            self.model = GPEN(
                model_path=self.model_path,
                size=512,  # 根据模型调整
                device=self.device
            )
            print("GPEN模型加载成功")
            
        except ImportError:
            print("无法导入GPEN，请安装相应的依赖")
            raise
    
    def enhance(self, img: np.ndarray, landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        """
        增强面部图像
        
        参数:
            img (np.ndarray): 输入BGR格式图像
            landmarks (np.ndarray, optional): 面部关键点，用于嘴唇保护
            
        返回:
            np.ndarray: 增强后的图像
        """
        if self.model is None:
            warnings.warn("面部增强模型未加载成功，返回原始图像")
            return img
        
        # 保存原始尺寸用于后续恢复
        original_height, original_width = img.shape[:2]
        
        # 预处理图像 - 确保是BGR格式并且是uint8类型
        processed_img = img.copy()
        if processed_img.dtype != np.uint8:
            processed_img = (np.clip(processed_img, 0, 1) * 255).astype(np.uint8)
        
        # 分离嘴唇区域（如果启用了嘴唇保护并提供了关键点）
        mouth_mask = None
        lips_img = None
        if self.mouth_protection and landmarks is not None:
            mouth_mask = self._create_mouth_mask(processed_img, landmarks)
            lips_img = cv2.bitwise_and(processed_img, processed_img, mask=mouth_mask)
            
        # 根据不同方法进行增强
        if self.method == 'gfpgan':
            enhanced_img = self._enhance_with_gfpgan(processed_img)
        elif self.method == 'codeformer':
            enhanced_img = self._enhance_with_codeformer(processed_img)
        elif self.method == 'gpen':
            enhanced_img = self._enhance_with_gpen(processed_img)
        else:
            enhanced_img = processed_img  # 如果方法无效，返回原始图像
        
        # 确保增强后的图像与输入尺寸一致
        if enhanced_img.shape[:2] != (original_height, original_width):
            enhanced_img = cv2.resize(
                enhanced_img,
                (original_width, original_height),
                interpolation=cv2.INTER_LANCZOS4
            )
        
        # 根据增强强度混合原始图像和增强图像
        if self.enhancement_strength < 1.0:
            enhanced_img = cv2.addWeighted(
                processed_img, 1 - self.enhancement_strength,
                enhanced_img, self.enhancement_strength,
                0
            )
        
        # 恢复嘴唇区域（如果启用了嘴唇保护）
        if self.mouth_protection and lips_img is not None and mouth_mask is not None:
            # 对嘴唇区域应用平滑过渡
            dilated_mask = cv2.dilate(mouth_mask, np.ones((5, 5), np.uint8), iterations=2)
            blurred_mask = cv2.GaussianBlur(dilated_mask, (15, 15), 0)
            blurred_mask = blurred_mask / 255.0 if blurred_mask.max() > 1.0 else blurred_mask
            
            # 根据嘴唇保护强度调整混合比例
            # 当mouth_protection_strength=0时，完全保留原始嘴唇
            # 当mouth_protection_strength=1时，完全使用增强后的嘴唇
            # 中间值则按比例混合
            if self.mouth_protection_strength > 0:
                # 调整掩码强度
                blurred_mask = blurred_mask * (1.0 - self.mouth_protection_strength)
            
            # 将浮点掩码转换回uint8格式
            if enhanced_img.dtype == np.uint8:
                blurred_mask = (blurred_mask * 255).astype(np.uint8)
            
            # 反转掩码
            inv_mask = cv2.bitwise_not(blurred_mask)
            
            # 准备增强图像的非嘴唇部分
            enhanced_non_lips = cv2.bitwise_and(enhanced_img, enhanced_img, mask=inv_mask)
            
            # 合并嘴唇和非嘴唇部分
            result = cv2.add(enhanced_non_lips, lips_img)
        else:
            result = enhanced_img
        
        return result
    
    def _enhance_with_gfpgan(self, img: np.ndarray) -> np.ndarray:
        """使用GFPGAN增强图像"""
        try:
            # 确保输入是BGR格式的uint8图像
            input_img = img.copy()
            
            # GFPGAN处理
            _, _, enhanced_img = self.model.enhance(
                input_img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )
            
            return enhanced_img
            
        except Exception as e:
            print(f"GFPGAN增强过程出错: {str(e)}")
            return img
    
    def _enhance_with_codeformer(self, img: np.ndarray) -> np.ndarray:
        """使用CodeFormer增强图像"""
        try:
            # 根据实际的CodeFormer API调整
            # 这里假设CodeFormer接口与GFPGAN类似
            enhanced_img = self.model.restore(
                img,
                w=self.enhancement_strength,  # CodeFormer特有的权重参数
                has_aligned=False,
                only_center_face=False
            )
            
            return enhanced_img
            
        except Exception as e:
            print(f"CodeFormer增强过程出错: {str(e)}")
            return img
    
    def _enhance_with_gpen(self, img: np.ndarray) -> np.ndarray:
        """使用GPEN增强图像"""
        try:
            # 根据实际的GPEN API调整处理流程
            enhanced_img = self.model.process(img)
            return enhanced_img
            
        except Exception as e:
            print(f"GPEN增强过程出错: {str(e)}")
            return img
    
    def _create_mouth_mask(self, img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        创建嘴唇区域的掩码
        
        参数:
            img (np.ndarray): 输入图像
            landmarks (np.ndarray): 面部关键点
            
        返回:
            np.ndarray: 嘴唇区域的掩码
        """
        height, width = img.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 假设landmarks包含面部68个关键点，其中49-68是嘴唇区域
        # 根据实际关键点格式调整索引
        try:
            mouth_pts = landmarks[48:68].astype(np.int32)
            cv2.fillPoly(mask, [mouth_pts], 255)
            
            # 扩大嘴唇区域，确保完全覆盖
            mask = cv2.dilate(mask, np.ones((10, 10), np.uint8), iterations=1)
        except Exception as e:
            print(f"创建嘴唇掩码出错: {str(e)}")
            # 如果出错，返回空掩码
            pass
        
        return mask 