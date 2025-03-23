import numpy as np
import cv2

class LipMaskGenerator:
    def __init__(self, blur_size=15, expansion_ratio=1.5):
        """
        初始化唇形蒙版生成器
        Args:
            blur_size: 高斯模糊的大小，控制边缘过渡的平滑程度
            expansion_ratio: 扩展比例，控制蒙版区域的大小
        """
        self.blur_size = blur_size
        self.expansion_ratio = expansion_ratio
        
        # 定义唇形相关的关键点索引
        self.outer_lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
        self.inner_lip_indices = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
        self.jaw_indices = [132, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]
        
    def create_lip_mask(self, landmarks, image_shape):
        """
        创建唇形区域的蒙版
        Args:
            landmarks: 人脸关键点坐标
            image_shape: 图像形状 (height, width)
        Returns:
            mask: 生成的蒙版
        """
        # 创建空白蒙版
        mask = np.zeros(image_shape[:2], dtype=np.float32)
        
        # 获取唇形区域的关键点
        outer_lip_points = landmarks[self.outer_lip_indices]
        inner_lip_points = landmarks[self.inner_lip_indices]
        jaw_points = landmarks[self.jaw_indices]
        
        # 计算唇形中心
        lip_center = np.mean(outer_lip_points, axis=0)
        
        # 计算唇形区域的大小
        lip_width = np.max(outer_lip_points[:, 0]) - np.min(outer_lip_points[:, 0])
        lip_height = np.max(outer_lip_points[:, 1]) - np.min(outer_lip_points[:, 1])
        
        # 扩展区域
        expanded_width = lip_width * self.expansion_ratio
        expanded_height = lip_height * self.expansion_ratio
        
        # 创建椭圆蒙版
        angle = 0  # 椭圆旋转角度
        center = tuple(map(int, lip_center))
        axes = (int(expanded_width/2), int(expanded_height/2))
        cv2.ellipse(mask, center, axes, angle, 0, 360, 1, -1)
        
        # 添加下巴区域
        jaw_points = jaw_points.astype(np.int32)
        cv2.fillPoly(mask, [jaw_points], 1)
        
        # 应用高斯模糊创建渐变边缘
        mask = cv2.GaussianBlur(mask, (self.blur_size, self.blur_size), 0)
        
        return mask
        
    def blend_images(self, original_img, modified_img, mask):
        """
        使用生成的蒙版混合原始图像和修改后的图像
        Args:
            original_img: 原始图像
            modified_img: 修改后的图像
            mask: 混合蒙版
        Returns:
            blended_img: 混合后的图像
        """
        # 确保mask是3通道的
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        
        # 使用蒙版混合图像
        blended_img = modified_img * mask + original_img * (1 - mask)
        return blended_img.astype(np.uint8) 