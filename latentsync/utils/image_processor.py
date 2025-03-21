# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torchvision import transforms
import cv2
from einops import rearrange
import mediapipe as mp
import torch
import numpy as np
from typing import Union
from .affine_transform import AlignRestore, laplacianSmooth
import face_alignment
import os

"""
If you are enlarging the image, you should prefer to use INTER_LINEAR or INTER_CUBIC interpolation. If you are shrinking the image, you should prefer to use INTER_AREA interpolation.
https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
"""


def load_fixed_mask(resolution: int, mask_image_path="latentsync/utils/mask.png") -> torch.Tensor:
    mask_image = cv2.imread(mask_image_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.resize(mask_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4) / 255.0
    mask_image = rearrange(torch.from_numpy(mask_image), "h w c -> c h w")
    return mask_image


class ImageProcessor:
    def __init__(self, resolution: int = 512, mask: str = "fix_mask", device: str = "cpu", mask_image=None):
        self.resolution = resolution
        self.resize = transforms.Resize(
            (resolution, resolution), 
            interpolation=transforms.InterpolationMode.LANCZOS, 
            antialias=True
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)
        self.mask = mask

        if mask in ["mouth", "face", "eye"]:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)  # Process single image
        if mask == "fix_mask":
            self.face_mesh = None
            self.smoother = laplacianSmooth()
            self.restorer = AlignRestore(upscale_factor=1.0)

            if mask_image is None:
                self.mask_image = load_fixed_mask(resolution)
            else:
                self.mask_image = mask_image

            if device != "cpu":
                self.fa = face_alignment.FaceAlignment(
                    face_alignment.LandmarksType.TWO_D, flip_input=False, device=device
                )
                self.face_mesh = None
            else:
                # self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)  # Process single image
                self.face_mesh = None
                self.fa = None

    def detect_facial_landmarks(self, image: np.ndarray):
        height, width, _ = image.shape
        results = self.face_mesh.process(image)
        if not results.multi_face_landmarks:  # Face not detected
            raise RuntimeError("Face not detected")
        face_landmarks = results.multi_face_landmarks[0]  # Only use the first face in the image
        landmark_coordinates = [
            (int(landmark.x * width), int(landmark.y * height)) for landmark in face_landmarks.landmark
        ]  # x means width, y means height
        return landmark_coordinates

    def preprocess_one_masked_image(self, image: torch.Tensor) -> np.ndarray:
        image = self.resize(image)

        if self.mask == "mouth" or self.mask == "face":
            landmark_coordinates = self.detect_facial_landmarks(image)
            if self.mask == "mouth":
                surround_landmarks = mouth_surround_landmarks
            else:
                surround_landmarks = face_surround_landmarks

            points = [landmark_coordinates[landmark] for landmark in surround_landmarks]
            points = np.array(points)
            mask = np.ones((self.resolution, self.resolution))
            mask = cv2.fillPoly(mask, pts=[points], color=(0, 0, 0))
            mask = torch.from_numpy(mask)
            mask = mask.unsqueeze(0)
        elif self.mask == "half":
            mask = torch.ones((self.resolution, self.resolution))
            height = mask.shape[0]
            mask[height // 2 :, :] = 0
            mask = mask.unsqueeze(0)
        elif self.mask == "eye":
            mask = torch.ones((self.resolution, self.resolution))
            landmark_coordinates = self.detect_facial_landmarks(image)
            y = landmark_coordinates[195][1]
            mask[y:, :] = 0
            mask = mask.unsqueeze(0)
        else:
            raise ValueError("Invalid mask type")

        image = image.to(dtype=torch.float32)
        pixel_values = self.normalize(image / 255.0)
        masked_pixel_values = pixel_values * mask
        mask = 1 - mask

        return pixel_values, masked_pixel_values, mask

    def affine_transform(self, image: torch.Tensor, allow_multi_faces: bool = True) -> np.ndarray:
        """
        对输入图像进行人脸对齐变换
        
        Args:
            image: 输入图像
            allow_multi_faces: 是否允许多个人脸
            
        Returns:
            face: 对齐后的人脸，形状为 [3, H, W]
            box: 人脸框，形状为 [4]
            affine_matrix: 仿射变换矩阵，形状为 [2, 3]
        """
        # 创建调试目录
        debug_dir = os.path.join(os.getcwd(), "debug_affine_steps")
        os.makedirs(debug_dir, exist_ok=True)
        
        # 生成唯一标识符
        import datetime
        import uuid
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        debug_prefix = f"{timestamp}_{unique_id}"
        
        # 保存输入图像用于调试
        input_path = os.path.join(debug_dir, f"{debug_prefix}_01_input.png")
        input_save = image.copy()
        
        # 分析输入图像的颜色分布
        print(f"===== 调试 affine_transform 过程 [{debug_prefix}] =====")
        print(f"输入图像 - 形状: {input_save.shape}, 类型: {input_save.dtype}")
        
        if len(input_save.shape) == 3 and input_save.shape[2] == 3:  # HWC格式
            b_mean, g_mean, r_mean = np.mean(input_save[:,:,0]), np.mean(input_save[:,:,1]), np.mean(input_save[:,:,2])
            print(f"输入图像 - 平均BGR值: B={b_mean:.2f}, G={g_mean:.2f}, R={r_mean:.2f}")
            input_color_space = "BGR" if b_mean > r_mean else "RGB"
            print(f"输入图像 - 推测颜色空间: {input_color_space}")
            
            # 保存输入图像
            try:
                cv2.imwrite(input_path, cv2.cvtColor(input_save, cv2.COLOR_RGB2BGR) if input_color_space == "RGB" else input_save)
                print(f"已保存输入图像到 {input_path}")
            except Exception as e:
                print(f"保存输入图像出错: {e}")
                
        # 原始affine_transform代码
        # image = rearrange(image, "c h w-> h w c").numpy()
        if self.fa is None:
            landmark_coordinates = np.array(self.detect_facial_landmarks(image))
            lm68 = mediapipe_lm478_to_face_alignment_lm68(landmark_coordinates)
        else:
            detected_faces = self.fa.get_landmarks(image)
            if detected_faces is None:
                raise RuntimeError("Face not detected")
            if not allow_multi_faces and len(detected_faces) > 1:
                raise RuntimeError("More than one face detected")
            lm68 = detected_faces[0]

        points = self.smoother.smooth(lm68)
        lmk3_ = np.zeros((3, 2))
        lmk3_[0] = points[17:22].mean(0)
        lmk3_[1] = points[22:27].mean(0)
        lmk3_[2] = points[27:36].mean(0)
        
        # 保存检测到的关键点
        landmarks_path = os.path.join(debug_dir, f"{debug_prefix}_02_landmarks.png")
        try:
            # 绘制关键点
            landmarks_vis = input_save.copy()
            for p in lm68:
                x, y = int(p[0]), int(p[1])
                cv2.circle(landmarks_vis, (x, y), 2, (0, 255, 0), -1)
            
            # 特别标记用于affine变换的3个关键点
            for i, p in enumerate(lmk3_):
                x, y = int(p[0]), int(p[1])
                cv2.circle(landmarks_vis, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(landmarks_vis, f"{i}", (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imwrite(landmarks_path, cv2.cvtColor(landmarks_vis, cv2.COLOR_RGB2BGR) if input_color_space == "RGB" else landmarks_vis)
            print(f"已保存关键点图像到 {landmarks_path}")
        except Exception as e:
            print(f"保存关键点图像出错: {e}")
        
        # 调用align_warp_face前保存图像的副本
        pre_warp_image = image.copy()
        
        # 执行warp_face操作
        print(f"执行 align_warp_face - 输入图像形状: {image.shape}")
        face, affine_matrix = self.restorer.align_warp_face(
            image.copy(), lmks3=lmk3_, smooth=True, border_mode="constant"
        )
        
        # 保存warp后的人脸
        warped_path = os.path.join(debug_dir, f"{debug_prefix}_03_warped_face.png")
        try:
            # 分析warp后的人脸颜色分布
            print(f"Warp后的人脸 - 形状: {face.shape}, 类型: {face.dtype}")
            
            if len(face.shape) == 3 and face.shape[2] == 3:
                b_mean, g_mean, r_mean = np.mean(face[:,:,0]), np.mean(face[:,:,1]), np.mean(face[:,:,2])
                print(f"Warp后的人脸 - 平均BGR值: B={b_mean:.2f}, G={g_mean:.2f}, R={r_mean:.2f}")
                face_color_space = "BGR" if b_mean > r_mean else "RGB"
                print(f"Warp后的人脸 - 推测颜色空间: {face_color_space}")
                
                # 保存为BGR格式以便OpenCV正确显示
                cv2.imwrite(warped_path, face if face_color_space == "BGR" else cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                print(f"已保存warp后的人脸到 {warped_path}")
        except Exception as e:
            print(f"保存warp后的人脸出错: {e}")
            
        box = [0, 0, face.shape[1], face.shape[0]]  # x1, y1, x2, y2
        
        # 调整大小前保存
        before_resize_path = os.path.join(debug_dir, f"{debug_prefix}_04_before_resize.png")
        try:
            cv2.imwrite(before_resize_path, face if face_color_space == "BGR" else cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            print(f"已保存调整大小前的图像到 {before_resize_path}")
        except Exception as e:
            print(f"保存调整大小前的图像出错: {e}")
        
        # 执行调整大小操作
        print(f"执行调整大小 - 输入大小: {face.shape[:2]} -> 目标大小: ({self.resolution}, {self.resolution})")
        face_resized = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
        
        # 保存调整大小后的图像
        resized_path = os.path.join(debug_dir, f"{debug_prefix}_05_resized.png")
        try:
            # 分析调整大小后的图像颜色分布
            print(f"调整大小后的图像 - 形状: {face_resized.shape}, 类型: {face_resized.dtype}")
            
            if len(face_resized.shape) == 3 and face_resized.shape[2] == 3:
                b_mean, g_mean, r_mean = np.mean(face_resized[:,:,0]), np.mean(face_resized[:,:,1]), np.mean(face_resized[:,:,2])
                print(f"调整大小后的图像 - 平均BGR值: B={b_mean:.2f}, G={g_mean:.2f}, R={r_mean:.2f}")
                resized_color_space = "BGR" if b_mean > r_mean else "RGB"
                print(f"调整大小后的图像 - 推测颜色空间: {resized_color_space}")
                
                # 保存为BGR格式以便OpenCV正确显示
                cv2.imwrite(resized_path, face_resized if resized_color_space == "BGR" else cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
                print(f"已保存调整大小后的图像到 {resized_path}")
        except Exception as e:
            print(f"保存调整大小后的图像出错: {e}")
            
        # 执行通道重排操作
        face_tensor = rearrange(torch.from_numpy(face_resized), "h w c -> c h w")
        
        # 保存最终输出的信息
        print(f"最终输出 - 形状: {face_tensor.shape}, 类型: {face_tensor.dtype}")
        print(f"===== 结束 affine_transform 调试 [{debug_prefix}] =====")
        
        return face_tensor, box, affine_matrix

    def preprocess_fixed_mask_image(self, image: torch.Tensor, affine_transform=False):
        if affine_transform:
            image, _, _ = self.affine_transform(image)
        else:
            image = self.resize(image)
        pixel_values = self.normalize(image / 255.0)
        masked_pixel_values = pixel_values * self.mask_image
        return pixel_values, masked_pixel_values, self.mask_image[0:1]

    def prepare_masks_and_masked_images(self, images: Union[torch.Tensor, np.ndarray], affine_transform=False):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")
        if self.mask == "fix_mask":
            results = [self.preprocess_fixed_mask_image(image, affine_transform=affine_transform) for image in images]
        else:
            results = [self.preprocess_one_masked_image(image) for image in images]

        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)

    def process_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")
        images = self.resize(images)
        pixel_values = self.normalize(images / 255.0)
        return pixel_values

    def close(self):
        if self.face_mesh is not None:
            self.face_mesh.close()


def mediapipe_lm478_to_face_alignment_lm68(lm478, return_2d=True):
    """
    lm478: [B, 478, 3] or [478,3]
    """
    # lm478[..., 0] *= W
    # lm478[..., 1] *= H
    landmarks_extracted = []
    for index in landmark_points_68:
        x = lm478[index][0]
        y = lm478[index][1]
        landmarks_extracted.append((x, y))
    return np.array(landmarks_extracted)


landmark_points_68 = [
    162,
    234,
    93,
    58,
    172,
    136,
    149,
    148,
    152,
    377,
    378,
    365,
    397,
    288,
    323,
    454,
    389,
    71,
    63,
    105,
    66,
    107,
    336,
    296,
    334,
    293,
    301,
    168,
    197,
    5,
    4,
    75,
    97,
    2,
    326,
    305,
    33,
    160,
    158,
    133,
    153,
    144,
    362,
    385,
    387,
    263,
    373,
    380,
    61,
    39,
    37,
    0,
    267,
    269,
    291,
    405,
    314,
    17,
    84,
    181,
    78,
    82,
    13,
    312,
    308,
    317,
    14,
    87,
]


# Refer to https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
mouth_surround_landmarks = [
    164,
    165,
    167,
    92,
    186,
    57,
    43,
    106,
    182,
    83,
    18,
    313,
    406,
    335,
    273,
    287,
    410,
    322,
    391,
    393,
]

face_surround_landmarks = [
    152,
    377,
    400,
    378,
    379,
    365,
    397,
    288,
    435,
    433,
    411,
    425,
    423,
    327,
    326,
    94,
    97,
    98,
    203,
    205,
    187,
    213,
    215,
    58,
    172,
    136,
    150,
    149,
    176,
    148,
]

if __name__ == "__main__":
    image_processor = ImageProcessor(512, mask="fix_mask")
    video = cv2.VideoCapture("assets/demo1_video.mp4")
    while True:
        ret, frame = video.read()
        # if not ret:
        #     break

        # cv2.imwrite("image.jpg", frame)

        frame = rearrange(torch.Tensor(frame).type(torch.uint8), "h w c ->  c h w")
        # face, masked_face, _ = image_processor.preprocess_fixed_mask_image(frame, affine_transform=True)
        face, _, _ = image_processor.affine_transform(frame)

        break

    face = (rearrange(face, "c h w -> h w c").detach().cpu().numpy()).astype(np.uint8)
    cv2.imwrite("face.jpg", face)

    # masked_face = (rearrange(masked_face, "c h w -> h w c").detach().cpu().numpy()).astype(np.uint8)
    # cv2.imwrite("masked_face.jpg", masked_face)
