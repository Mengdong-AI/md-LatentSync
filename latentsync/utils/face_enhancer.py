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
        model_path: Optional[str] = None,
        enable: bool = True
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
            enable (bool): 是否启用面部增强
        """
        self.method = method.lower()
        self.enhancement_strength = enhancement_strength
        self.upscale = upscale
        self.device = device
        self.mouth_protection = mouth_protection
        self.mouth_protection_strength = max(0.0, min(1.0, mouth_protection_strength))
        self.model = None
        self.model_path = model_path
        self.enable = enable
        
        if not self.enable:
            print("面部增强已禁用")
            return
        
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
            print("无法导入GFPGAN，尝试备选导入方式...")
            try:
                import sys
                import os
                # 检查是否有GFPGAN目录，如果有则添加到路径
                if os.path.exists('GFPGAN'):
                    sys.path.append('GFPGAN')
                from gfpgan.gfpgan import GFPGANer
                
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
                print("GFPGAN模型加载成功（备选方法）")
            except Exception as e:
                print(f"无法导入GFPGAN: {str(e)}")
                print("请安装相应的依赖: pip install git+https://github.com/TencentARC/GFPGAN.git")
                self.enable = False
    
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
            print("无法导入CodeFormer，尝试备选导入方式...")
            try:
                import sys
                # 检查是否有CodeFormer目录
                if os.path.exists('CodeFormer'):
                    sys.path.append('CodeFormer')
                    from codeformer.basicsr.archs.codeformer_arch import CodeFormer
                    from codeformer.basicsr.utils import img2tensor, tensor2img
                    from codeformer.basicsr.utils.registry import ARCH_REGISTRY
                    from torchvision.transforms.functional import normalize
                    
                    # 创建一个简单的包装类来模拟CodeFormerRestorer
                    class SimpleCodeFormerRestorer:
                        def __init__(self, model_path, upscale, device):
                            self.device = device
                            self.upscale = upscale
                            self.model = ARCH_REGISTRY.get('CodeFormer')(
                                dim_embd=512,
                                codebook_size=1024,
                                n_head=8,
                                n_layers=9,
                                connect_list=['32', '64', '128', '256']
                            ).to(device)
                            
                            # 加载预训练模型
                            checkpoint = torch.load(model_path, map_location=device)
                            self.model.load_state_dict(checkpoint['params_ema'])
                            self.model.eval()
                            
                        def restore(self, img, w=0.5, has_aligned=False, only_center_face=False):
                            with torch.no_grad():
                                # 预处理
                                img_tensor = img2tensor(img, bgr2rgb=False, float32=True)
                                normalize(img_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                                
                                # 模型推理
                                output = self.model(img_tensor, w=w)
                                
                                # 后处理
                                restored_img = tensor2img(output, rgb2bgr=False, min_max=(-1, 1))
                                return restored_img
                    
                    # 设定模型路径
                    if self.model_path is None:
                        self.model_path = 'models/faceenhancer/codeformer.pth'
                        
                        # 如果模型不存在，打印下载说明
                        if not os.path.exists(self.model_path):
                            print(f"CodeFormer模型不存在: {self.model_path}")
                            print("请从 https://github.com/sczhou/CodeFormer/releases 下载模型")
                            print(f"并将其放置到 {self.model_path}")
                            raise FileNotFoundError(f"CodeFormer模型不存在: {self.model_path}")
                    
                    # 初始化简单的CodeFormer包装类
                    self.model = SimpleCodeFormerRestorer(
                        model_path=self.model_path,
                        upscale=self.upscale,
                        device=self.device
                    )
                    print("CodeFormer模型加载成功（备选方法）")
                else:
                    raise ImportError("CodeFormer目录不存在，请克隆仓库: git clone https://github.com/sczhou/CodeFormer.git")
            except Exception as e:
                print(f"无法导入CodeFormer: {str(e)}")
                print("请安装相应的依赖")
                self.enable = False
    
    def _load_gpen(self):
        """加载GPEN模型"""
        try:
            import sys
            import os
            
            # 检查模型文件是否存在
            if self.model_path is None:
                self.model_path = 'models/faceenhancer/GPEN-BFR-512.pth'
                
                # 如果模型不存在，打印下载说明
                if not os.path.exists(self.model_path):
                    print(f"GPEN模型不存在: {self.model_path}")
                    print("请从 https://github.com/yangxy/GPEN/releases 下载模型")
                    print(f"并将其放置到 {self.model_path}")
                    raise FileNotFoundError(f"GPEN模型不存在: {self.model_path}")
            
            # 直接实现简单的 GPEN 处理类
            class SimpleGPEN:
                def __init__(self, model_path, size=512, device='cuda'):
                    self.model_path = model_path
                    self.size = size
                    self.device = device
                    print(f"初始化GPEN模型: {model_path}, 尺寸: {size}, 设备: {device}")
                    # 在实际使用时才导入依赖模块并加载模型
                
                def process(self, img):
                    """处理图像"""
                    # 由于没有实际的GPEN模型，这里先返回原图
                    print(f"注意: GPEN模型尚未实际加载，返回原始图像")
                    return img
            
            # 初始化简单GPEN类
            self.model = SimpleGPEN(
                model_path=self.model_path,
                size=512,
                device=self.device
            )
            print("GPEN模型准备就绪（简化版）")
            print("注意：当前实现将返回未增强的原始图像")
            
        except Exception as e:
            print(f"加载GPEN模型失败: {str(e)}")
            print("GPEN增强器将返回未处理的原始图像")
            self.model = None
            self.enable = False
    
    def enhance(self, img: np.ndarray, landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        """
        增强面部图像
        
        参数:
            img (np.ndarray): 输入BGR格式图像
            landmarks (np.ndarray, optional): 面部关键点，用于嘴唇保护
            
        返回:
            np.ndarray: 增强后的图像
        """
        if not self.enable or self.model is None:
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
        elif self.mouth_protection:
            # 如果启用了嘴唇保护但没有提供关键点，则使用简单的固定区域作为嘴唇区域
            h, w = processed_img.shape[:2]
            mouth_mask = np.zeros((h, w), dtype=np.uint8)
            # 估计嘴唇位置在下半部分中间区域
            mouth_top = int(h * 0.6)
            mouth_bottom = int(h * 0.85)
            mouth_left = int(w * 0.3)
            mouth_right = int(w * 0.7)
            # 填充嘴唇区域
            mouth_mask[mouth_top:mouth_bottom, mouth_left:mouth_right] = 255
            # 平滑过渡边缘
            mouth_mask = cv2.GaussianBlur(mouth_mask, (31, 31), 0)
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
            # 根据嘴唇保护强度调整混合比例
            if mouth_mask.max() > 1.0:  # 如果是0-255范围
                mouth_mask = mouth_mask / 255.0
            
            # 当mouth_protection_strength=0时，完全保留原始嘴唇
            # 当mouth_protection_strength=1时，完全使用增强后的嘴唇
            if self.mouth_protection_strength > 0:
                mouth_mask = mouth_mask * (1.0 - self.mouth_protection_strength)
            
            # 混合嘴唇区域
            blended = enhanced_img.copy().astype(np.float32)
            for c in range(3):  # 对每个颜色通道
                blended[:,:,c] = enhanced_img[:,:,c] * (1 - mouth_mask) + processed_img[:,:,c] * mouth_mask
            
            enhanced_img = blended.astype(np.uint8)
        
        return enhanced_img
    
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
        if self.model is None:
            return img
            
        try:
            # 尝试使用GPEN处理图像
            enhanced_img = self.model.process(img)
            if enhanced_img is None or enhanced_img.shape != img.shape:
                print("GPEN处理返回无效图像，返回原始图像")
                return img
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