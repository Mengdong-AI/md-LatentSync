# Adapted from https://github.com/guanjz20/StyleSync/blob/main/utils.py

import numpy as np
import cv2
import torch


def transformation_from_points(points1, points0, smooth=True, p_bias=None):
    points2 = np.array(points0)
    points2 = points2.astype(np.float64)
    points1 = points1.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(np.matmul(points1.T, points2))
    R = (np.matmul(U, Vt)).T
    sR = (s2 / s1) * R
    T = c2.reshape(2, 1) - (s2 / s1) * np.matmul(R, c1.reshape(2, 1))
    M = np.concatenate((sR, T), axis=1)
    if smooth:
        bias = points2[2] - points1[2]
        if p_bias is None:
            p_bias = bias
        else:
            bias = p_bias * 0.2 + bias * 0.8
        p_bias = bias
        M[:, 2] = M[:, 2] + bias
    return M, p_bias


class AlignRestore:
    def __init__(self, align_points=3, upscale_factor=1.0):
        if align_points == 3:
            self.upscale_factor = upscale_factor
            ratio = 2.8
            self.crop_ratio = (ratio, ratio)
            self.face_template = np.array([[19 - 2, 30 - 10], [56 + 2, 30 - 10], [37.5, 45 - 5]])
            self.face_template = self.face_template * ratio
            self.face_size = (int(75 * self.crop_ratio[0]), int(100 * self.crop_ratio[1]))
            self.p_bias = None

    def process(self, img, lmk_align=None, smooth=True, align_points=3):
        aligned_face, affine_matrix = self.align_warp_face(img, lmk_align, smooth)
        restored_img = self.restore_img(img, aligned_face, affine_matrix)
        cv2.imwrite("restored.jpg", restored_img)
        cv2.imwrite("aligned.jpg", aligned_face)
        return aligned_face, restored_img

    def align_warp_face(self, img, lmks3, smooth=True, border_mode="constant"):
        affine_matrix, self.p_bias = transformation_from_points(lmks3, self.face_template, smooth, self.p_bias)
        if border_mode == "constant":
            border_mode = cv2.BORDER_CONSTANT
        elif border_mode == "reflect101":
            border_mode = cv2.BORDER_REFLECT101
        elif border_mode == "reflect":
            border_mode = cv2.BORDER_REFLECT

        cropped_face = cv2.warpAffine(
            img,
            affine_matrix,
            self.face_size,
            flags=cv2.INTER_LANCZOS4,
            borderMode=border_mode,
            borderValue=[127, 127, 127],
        )
        return cropped_face, affine_matrix

    def align_warp_face2(self, img, landmark, border_mode="constant"):
        affine_matrix = cv2.estimateAffinePartial2D(landmark, self.face_template)[0]
        if border_mode == "constant":
            border_mode = cv2.BORDER_CONSTANT
        elif border_mode == "reflect101":
            border_mode = cv2.BORDER_REFLECT101
        elif border_mode == "reflect":
            border_mode = cv2.BORDER_REFLECT
        cropped_face = cv2.warpAffine(
            img, affine_matrix, self.face_size, borderMode=border_mode, borderValue=(135, 133, 132)
        )
        return cropped_face, affine_matrix

    def restore_img(self, img, face, affine_matrix):
        """
        将生成的人脸图像粘贴回原始图像
        Args:
            img: 原始图像
            face: 生成的人脸图像
            affine_matrix: 仿射变换矩阵
        Returns:
            粘贴后的图像
        """
        # 确保 face 是 uint8 类型
        if isinstance(face, torch.Tensor):
            face = face.cpu().numpy()
        if face.dtype != np.uint8:
            face = (face * 255).astype(np.uint8)

        # 获取原始图像尺寸
        h, w = img.shape[:2]
        
        # 根据 upscale_factor 调整仿射矩阵
        if self.upscale_factor != 1.0:
            affine_matrix = affine_matrix.copy()
            affine_matrix[:, 2] *= self.upscale_factor
            affine_matrix[:, :2] *= self.upscale_factor
        
        # 创建掩码
        mask = np.ones_like(face, dtype=np.uint8) * 255
        
        # 使用仿射变换将人脸和掩码变换回原始图像空间
        warped_face = cv2.warpAffine(face, affine_matrix, (w, h), flags=cv2.INTER_LANCZOS4)
        warped_mask = cv2.warpAffine(mask, affine_matrix, (w, h), flags=cv2.INTER_LANCZOS4)
        
        # 对掩码进行腐蚀操作，创建平滑过渡
        kernel_size = max(1, int(min(h, w) * 0.02))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        warped_mask = cv2.erode(warped_mask, kernel, iterations=1)
        
        # 创建软掩码，确保维度匹配
        soft_mask = warped_mask.astype(np.float32) / 255.0
        if len(soft_mask.shape) == 2:  # 如果是单通道，扩展为三通道
            soft_mask = np.expand_dims(soft_mask, axis=-1)
        
        # 混合图像
        result = img * (1 - soft_mask) + warped_face * soft_mask
        
        # 确保输出类型正确
        max_value = np.iinfo(img.dtype).max if img.dtype != np.float32 else 1.0
        result = np.clip(result, 0, max_value).astype(img.dtype)
        
        return result


class laplacianSmooth:
    def __init__(self, smoothAlpha=0.3):
        self.smoothAlpha = smoothAlpha
        self.pts_last = None

    def smooth(self, pts_cur):
        if self.pts_last is None:
            self.pts_last = pts_cur.copy()
            return pts_cur.copy()
        x1 = min(pts_cur[:, 0])
        x2 = max(pts_cur[:, 0])
        y1 = min(pts_cur[:, 1])
        y2 = max(pts_cur[:, 1])
        width = x2 - x1
        pts_update = []
        for i in range(len(pts_cur)):
            x_new, y_new = pts_cur[i]
            x_old, y_old = self.pts_last[i]
            tmp = (x_new - x_old) ** 2 + (y_new - y_old) ** 2
            w = np.exp(-tmp / (width * self.smoothAlpha))
            x = x_old * w + x_new * (1 - w)
            y = y_old * w + y_new * (1 - w)
            pts_update.append([x, y])
        pts_update = np.array(pts_update)
        self.pts_last = pts_update.copy()

        return pts_update
