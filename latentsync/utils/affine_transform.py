# Adapted from https://github.com/guanjz20/StyleSync/blob/main/utils.py

import numpy as np
import cv2
import os
import datetime
import uuid


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


class AlignRestore(object):
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
        """
        对人脸图像进行仿射变换对齐
        
        Args:
            img: 输入图像
            lmks3: 3个关键点坐标
            smooth: 是否平滑变换
            border_mode: 边界模式
            
        Returns:
            cropped_face: 变换后的人脸图像
            affine_matrix: 仿射变换矩阵
        """
        # 创建调试目录
        debug_dir = os.path.join(os.getcwd(), "debug_warp_steps")
        os.makedirs(debug_dir, exist_ok=True)
        
        # 生成唯一标识符
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        debug_prefix = f"{timestamp}_{unique_id}"
        
        # 保存输入图像用于调试
        input_path = os.path.join(debug_dir, f"{debug_prefix}_01_input.png")
        input_save = img.copy()
        
        # 分析输入图像的颜色分布
        print(f"===== 调试 align_warp_face 过程 [{debug_prefix}] =====")
        print(f"输入图像 - 形状: {input_save.shape}, 类型: {input_save.dtype}")
        
        if len(input_save.shape) == 3 and input_save.shape[2] == 3:
            b_mean, g_mean, r_mean = np.mean(input_save[:,:,0]), np.mean(input_save[:,:,1]), np.mean(input_save[:,:,2])
            print(f"输入图像 - 平均BGR值: B={b_mean:.2f}, G={g_mean:.2f}, R={r_mean:.2f}")
            input_color_space = "BGR" if b_mean > r_mean else "RGB"
            print(f"输入图像 - 推测颜色空间: {input_color_space}")
            
            # 保存输入图像
            cv2.imwrite(input_path, cv2.cvtColor(input_save, cv2.COLOR_RGB2BGR) if input_color_space == "RGB" else input_save)
            print(f"已保存输入图像到 {input_path}")
            
            # 可视化关键点
            landmarks_img = input_save.copy()
            for i, point in enumerate(lmks3):
                x, y = int(point[0]), int(point[1])
                cv2.circle(landmarks_img, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(landmarks_img, f"{i}", (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            # 保存带关键点的图像
            landmarks_path = os.path.join(debug_dir, f"{debug_prefix}_02_landmarks.png")
            cv2.imwrite(landmarks_path, cv2.cvtColor(landmarks_img, cv2.COLOR_RGB2BGR) if input_color_space == "RGB" else landmarks_img)
            print(f"已保存带关键点的图像到 {landmarks_path}")
            
            # 可视化模板关键点
            template_img = np.zeros((self.face_size[1], self.face_size[0], 3), dtype=np.uint8)
            for i, point in enumerate(self.face_template):
                x, y = int(point[0]), int(point[1])
                cv2.circle(template_img, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(template_img, f"{i}", (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            # 保存模板关键点图像
            template_path = os.path.join(debug_dir, f"{debug_prefix}_03_template.png")
            cv2.imwrite(template_path, template_img)
            print(f"已保存模板关键点图像到 {template_path}")
        
        # 计算仿射变换矩阵
        print(f"计算仿射变换矩阵 - 关键点: {lmks3}, 目标模板: {self.face_template}")
        affine_matrix, self.p_bias = transformation_from_points(lmks3, self.face_template, smooth, self.p_bias)
        print(f"计算得到的仿射变换矩阵: \n{affine_matrix}")
        
        # 设置边界模式
        if border_mode == "constant":
            border_mode_cv = cv2.BORDER_CONSTANT
            print(f"边界模式: BORDER_CONSTANT, 边界值: [127, 127, 127]")
        elif border_mode == "reflect101":
            border_mode_cv = cv2.BORDER_REFLECT101
            print(f"边界模式: BORDER_REFLECT101")
        elif border_mode == "reflect":
            border_mode_cv = cv2.BORDER_REFLECT
            print(f"边界模式: BORDER_REFLECT")
        else:
            border_mode_cv = cv2.BORDER_CONSTANT
            print(f"未知边界模式: {border_mode}, 使用默认值: BORDER_CONSTANT")

        # 执行仿射变换
        print(f"执行仿射变换 - 输入大小: {img.shape[:2]}, 目标大小: {self.face_size}")
        cropped_face = cv2.warpAffine(
            img,
            affine_matrix,
            self.face_size,
            flags=cv2.INTER_LANCZOS4,
            borderMode=border_mode_cv,
            borderValue=[127, 127, 127],
        )
        
        # 保存变换后的图像
        output_path = os.path.join(debug_dir, f"{debug_prefix}_04_warped.png")
        
        # 分析输出图像的颜色分布
        print(f"输出图像 - 形状: {cropped_face.shape}, 类型: {cropped_face.dtype}")
        
        if len(cropped_face.shape) == 3 and cropped_face.shape[2] == 3:
            b_mean, g_mean, r_mean = np.mean(cropped_face[:,:,0]), np.mean(cropped_face[:,:,1]), np.mean(cropped_face[:,:,2])
            print(f"输出图像 - 平均BGR值: B={b_mean:.2f}, G={g_mean:.2f}, R={r_mean:.2f}")
            output_color_space = "BGR" if b_mean > r_mean else "RGB"
            print(f"输出图像 - 推测颜色空间: {output_color_space}")
            
            # 检查颜色空间是否发生变化
            if input_color_space != output_color_space:
                print(f"警告: 颜色空间发生变化! 输入: {input_color_space} -> 输出: {output_color_space}")
            else:
                print(f"颜色空间保持一致: {input_color_space}")
            
            # 保存输出图像
            cv2.imwrite(output_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR) if output_color_space == "RGB" else cropped_face)
            print(f"已保存变换后的图像到 {output_path}")
        
        print(f"===== 结束 align_warp_face 调试 [{debug_prefix}] =====")
        
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

    def restore_img(self, input_img, face, affine_matrix, return_mask=False):
        h, w, _ = input_img.shape
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)
        
        if self.upscale_factor != 1.0:
            upsample_img = cv2.resize(input_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        else:
            upsample_img = input_img.copy()
            
        inverse_affine = cv2.invertAffineTransform(affine_matrix)
        if self.upscale_factor != 1.0:
            inverse_affine *= self.upscale_factor
            extra_offset = 0.5 * self.upscale_factor
        else:
            extra_offset = 0
        inverse_affine[:, 2] += extra_offset
        
        inv_restored = cv2.warpAffine(face, inverse_affine, (w_up, h_up), 
                                     flags=cv2.INTER_LANCZOS4,
                                     borderMode=cv2.BORDER_TRANSPARENT)
        
        mask = np.ones((self.face_size[1], self.face_size[0]), dtype=np.float32)
        inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up),
                                 flags=cv2.INTER_LANCZOS4)
        
        inv_mask_erosion = cv2.erode(
            inv_mask, np.ones((int(2 * self.upscale_factor), int(2 * self.upscale_factor)), np.uint8)
        )
        pasted_face = inv_mask_erosion[:, :, None] * inv_restored
        total_face_area = np.sum(inv_mask_erosion)
        w_edge = max(int(total_face_area**0.5) // 20, 1)
        erosion_radius = w_edge * 2
        inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
        
        blur_size = max(w_edge, 3)
        inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size*2+1, blur_size*2+1), 0)
        
        if inv_soft_mask.shape[0] != h or inv_soft_mask.shape[1] != w:
            if return_mask:
                inv_restored = cv2.resize(inv_restored, (w, h), interpolation=cv2.INTER_LANCZOS4)
                inv_soft_mask = cv2.resize(inv_soft_mask, (w, h), interpolation=cv2.INTER_LANCZOS4)
            else:
                inv_soft_mask = inv_soft_mask[:, :, None]
                upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img
                if self.upscale_factor != 1.0:
                    upsample_img = cv2.resize(upsample_img, (w, h), interpolation=cv2.INTER_LANCZOS4)
        else:
            if not return_mask:
                inv_soft_mask = inv_soft_mask[:, :, None]
            
        if return_mask:
            if len(inv_soft_mask.shape) > 2:
                inv_soft_mask = inv_soft_mask[:, :, 0] if inv_soft_mask.shape[2] > 0 else inv_soft_mask[:, :]
            return inv_restored, inv_soft_mask
        
        upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img
        
        if np.max(upsample_img) > 256:
            upsample_img = upsample_img.astype(np.uint16)
        else:
            upsample_img = upsample_img.astype(np.uint8)
        return upsample_img


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
