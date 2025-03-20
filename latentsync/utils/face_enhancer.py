import cv2
import numpy as np
import os

class FaceEnhancer:
    """
    面部增强器类 - 提供多种面部增强方法
    """
    def __init__(self, enhancement_strength=0.8, method='sharpen'):
        """
        初始化面部增强器
        
        Args:
            enhancement_strength: 增强强度，范围0.0-1.0，1.0表示完全增强，0.0表示保持原图
            method: 增强方法，可选 'sharpen', 'clahe', 'detail', 'combined'
        """
        self.enhancement_strength = max(0.0, min(1.0, enhancement_strength))
        self.method = method
    
    def enhance(self, face):
        """
        增强面部图像
        
        Args:
            face: 输入面部图像 (BGR格式)
            
        Returns:
            enhanced_face: 增强后的面部图像 (BGR格式)
        """
        # 保存原始图像
        original_face = face.copy()
        
        # 根据指定方法应用增强
        if self.method == 'sharpen':
            enhanced_face = self._sharpen(face)
        elif self.method == 'clahe':
            enhanced_face = self._apply_clahe(face)
        elif self.method == 'detail':
            enhanced_face = self._enhance_detail(face)
        elif self.method == 'combined':
            enhanced_face = self._combined_enhancement(face)
        else:
            enhanced_face = face.copy()
        
        # 根据增强强度混合原图和增强结果
        if self.enhancement_strength < 1.0:
            enhanced_face = cv2.addWeighted(
                enhanced_face, self.enhancement_strength,
                original_face, 1.0 - self.enhancement_strength, 0
            )
        
        return enhanced_face
    
    def _sharpen(self, face):
        """
        使用锐化滤镜增强面部细节
        """
        # 创建锐化滤镜
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        # 应用锐化
        sharpened = cv2.filter2D(face, -1, kernel)
        return sharpened
    
    def _apply_clahe(self, face):
        """
        使用CLAHE (对比度受限的自适应直方图均衡化) 增强面部
        """
        # 转换为LAB颜色空间
        lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        # 分离L通道
        l, a, b = cv2.split(lab)
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # 应用CLAHE到L通道
        enhanced_l = clahe.apply(l)
        # 合并通道
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        # 转换回BGR颜色空间
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    def _enhance_detail(self, face):
        """
        使用细节增强滤镜增强面部
        """
        # 使用双边滤波器保持边缘的同时降噪
        blurred = cv2.bilateralFilter(face, 9, 75, 75)
        # 获取细节层 (原图减去平滑图)
        detail = cv2.subtract(face, blurred)
        # 增强细节
        enhanced = cv2.addWeighted(face, 1.5, detail, 0.5, 0)
        return enhanced
    
    def _combined_enhancement(self, face):
        """
        组合多种方法增强面部
        """
        # 先应用CLAHE
        clahe_face = self._apply_clahe(face)
        # 再锐化
        sharpened = self._sharpen(clahe_face)
        # 最后应用细节增强
        enhanced = cv2.addWeighted(sharpened, 0.7, clahe_face, 0.3, 0)
        return enhanced 