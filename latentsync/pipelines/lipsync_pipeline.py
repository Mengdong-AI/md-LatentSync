# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import math
import os
import shutil
from typing import Callable, List, Optional, Union
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torchvision
from torchvision import transforms

from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange
import cv2

from ..models.unet import UNet3DConditionModel
from ..utils.util import read_video, read_audio, write_video, check_ffmpeg_installed
from ..utils.image_processor import ImageProcessor, load_fixed_mask
from ..whisper.audio2feature import Audio2Feature
import tqdm
import soundfile as sf

from ..utils.face_enhancer import FaceEnhancer
from ..utils.batch_face_enhancer import BatchFaceEnhancer

import mediapipe as mp
import face_alignment

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        denoising_unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(denoising_unet.config, "_diffusers_version") and version.parse(
            version.parse(denoising_unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = (
            hasattr(denoising_unet.config, "sample_size") and denoising_unet.config.sample_size < 64
        )
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(denoising_unet.config)
            new_config["sample_size"] = 64
            denoising_unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            denoising_unet=denoising_unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

        # Initialize face enhancer
        self.face_enhancer = FaceEnhancer(
            enhancement_method='gpen',
            device='cuda',
            enhancement_strength=0.5,
            enable=True
        )

        # 初始化批处理人脸增强器
        self.batch_face_enhancer = None
        self.face_enhancer_config = {
            'enhancement_method': 'gpen',
            'device': 'cuda',
            'enhancement_strength': 0.5,
            'enable': True,
            'num_workers': 2,
            'mouth_protection': True,
            'mouth_protection_strength': 0.8
        }

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.denoising_unet, "_hf_hook"):
            return self.device
        for module in self.denoising_unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            batch_size,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    def affine_transform_video(self, video_frames: np.ndarray):
        faces = []
        boxes = []
        affine_matrices = []
        face_landmarks = []  # 新增：存储人脸关键点
        print(f"Affine transforming {len(video_frames)} faces...")
        
        start_time = time.time()
        
        try:
            # 创建线程池
            num_workers = min(8, os.cpu_count() or 4)  # 最多8个线程
            batch_size = 16  # 每批处理16帧
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # 将视频帧分成批次
                batches = [video_frames[i:i + batch_size] for i in range(0, len(video_frames), batch_size)]
                futures = []
                
                # 提交批次任务
                for batch_idx, batch in enumerate(batches):
                    future = executor.submit(self._process_affine_batch, batch, batch_idx * batch_size)
                    futures.append(future)
                
                # 收集结果
                results = []
                for future in tqdm.tqdm(futures, desc="Processing batches"):
                    try:
                        batch_results = future.result()
                        results.extend(batch_results)
                            
                    except Exception as e:
                        print(f"处理批次时出错: {str(e)}")
                        # 如果是 CUDA OOM，尝试减小批大小重试
                        if "CUDA out of memory" in str(e):
                            print("CUDA 内存不足，尝试减小批大小重试")
                            batch_size = max(1, batch_size // 2)
                            continue
                
                # 按帧索引排序结果
                results.sort(key=lambda x: x[0])
                
                # 分离结果，处理空结果
                valid_results = 0
                for idx, face, box, affine_matrix, landmarks in results:  # 修改：添加 landmarks
                    if face is not None and box is not None and affine_matrix is not None:
                        faces.append(face)
                        boxes.append(box)
                        affine_matrices.append(affine_matrix)
                        face_landmarks.append(landmarks)  # 新增：添加关键点
                        valid_results += 1
                    else:
                        print(f"警告：第 {idx} 帧处理失败，将使用临近帧的结果")
                        if valid_results > 0:
                            # 使用最近的有效结果
                            faces.append(faces[-1])
                            boxes.append(boxes[-1])
                            affine_matrices.append(affine_matrices[-1])
                            face_landmarks.append(face_landmarks[-1])  # 新增：添加关键点
                        else:
                            # 如果没有有效结果，创建默认值
                            print(f"错误：没有有效的处理结果用于第 {idx} 帧")
                            raise RuntimeError(f"无法处理第 {idx} 帧")
            
            process_time = time.time() - start_time
            print(f"\nAffine Transform 处理时间统计:")
            print(f"总处理时间: {process_time:.2f}秒")
            print(f"平均每帧处理时间: {process_time/len(video_frames):.3f}秒")
            print(f"处理速度: {len(video_frames)/process_time:.1f} 帧/秒")
            print(f"使用线程数: {num_workers}")
            print(f"最终批大小: {batch_size}")
            print(f"成功处理帧数: {valid_results}/{len(video_frames)}")
            
            # 确保清理 CUDA 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            faces = torch.stack(faces)
            return faces, boxes, affine_matrices, face_landmarks  # 修改：返回关键点
            
        except Exception as e:
            print(f"Affine Transform 处理失败: {str(e)}")
            # 清理 CUDA 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    
    def _process_affine_batch(self, batch_frames, start_idx):
        """处理一批视频帧的 affine transform
        
        Args:
            batch_frames: 一批视频帧
            start_idx: 起始帧索引
            
        Returns:
            list of (frame_idx, face, box, affine_matrix, landmarks)
        """
        results = []
        try:
            for i, frame in enumerate(batch_frames):
                frame_idx = start_idx + i
                try:
                    face, box, affine_matrix, landmarks = self.image_processor.affine_transform(frame)
                    results.append((frame_idx, face, box, affine_matrix, landmarks))
                    
                    # 每处理一帧就清理一次 CUDA 缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"处理第 {frame_idx} 帧时出错: {str(e)}")
                    # 使用空结果作为占位符
                    results.append((frame_idx, None, None, None, None))
            
            return results
            
        except Exception as e:
            print(f"批处理失败: {str(e)}")
            # 确保清理 CUDA 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def restore_video(self, faces: torch.Tensor, video_frames: np.ndarray, boxes: list, affine_matrices: list, face_landmarks: list = None):
        """还原视频帧
        
        Args:
            faces: 处理后的人脸张量
            video_frames: 原始视频帧
            boxes: 人脸框列表
            affine_matrices: 仿射变换矩阵列表
            face_landmarks: 人脸关键点列表
        """
        video_frames = video_frames[: len(faces)]
        out_frames = []
        print(f"Restoring {len(faces)} faces...")
        
        start_time = time.time()
        
        # 首先处理所有人脸
        face_frames = []
        for index, face in enumerate(tqdm.tqdm(faces)):
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(face, size=(height, width), antialias=True)
            face = rearrange(face, "c h w -> h w c")
            face = (face / 2 + 0.5).clamp(0, 1)
            face = (face * 255).to(torch.uint8).cpu().numpy()
            
            # 检查输入人脸数据类型
            if face.dtype != np.uint8:
                print(f"Warning: Input face dtype is {face.dtype}, converting to uint8")
                face = np.clip(face, 0, 255).astype(np.uint8)
            
            face_frames.append(face)
        
        preprocess_time = time.time() - start_time
        print(f"预处理完成，耗时: {preprocess_time:.2f}秒")
        
        enhance_start_time = time.time()
        # 使用合适的方式处理人脸增强
        if self.batch_face_enhancer is not None:
            try:
                # 如果帧数较多且worker数足够，使用批处理
                if len(face_frames) > 50 and self.face_enhancer_config.get('num_workers', 2) >= 4:
                    print("使用批处理进行人脸增强...")
                    # 提交所有人脸进行处理
                    for i, face in enumerate(face_frames):
                        landmarks = face_landmarks[i] if face_landmarks is not None else None
                        self.batch_face_enhancer.process_frame(i, face, landmarks)
                    
                    # 收集增强结果
                    enhanced_faces = []
                    frame_indices = []
                    
                    while len(enhanced_faces) < len(face_frames):
                        try:
                            idx, enhanced_face = self.batch_face_enhancer.get_result(timeout=10.0)
                            if idx is not None and enhanced_face is not None:
                                enhanced_faces.append((idx, enhanced_face))
                                frame_indices.append(idx)
                                if len(enhanced_faces) % 10 == 0:
                                    print(f"已处理 {len(enhanced_faces)}/{len(face_frames)} 帧")
                        except Exception as e:
                            print(f"获取增强结果时出错: {str(e)}")
                            continue
                    
                    # 按帧索引排序结果
                    enhanced_faces.sort(key=lambda x: x[0])
                    face_frames = [face for _, face in enhanced_faces]
                else:
                    print("使用单线程进行人脸增强...")
                    # 直接使用 FaceEnhancer 处理
                    enhanced_faces = []
                    for i, face in enumerate(tqdm.tqdm(face_frames)):
                        try:
                            landmarks = face_landmarks[i] if face_landmarks is not None else None
                            enhanced_face = self.batch_face_enhancer.enhancers[0].enhance(face, landmarks)
                            if enhanced_face is None:
                                print(f"Warning: Face enhancement failed for frame {i}, using original face")
                                enhanced_face = face
                            enhanced_faces.append(enhanced_face)
                        except Exception as e:
                            print(f"处理第 {i} 帧时出错: {str(e)}")
                            enhanced_faces.append(face)
                    face_frames = enhanced_faces
                
            except Exception as e:
                print(f"人脸增强时出错: {str(e)}")
                # 保持原始人脸不变
                pass
        
        enhance_time = time.time() - enhance_start_time
        print(f"人脸增强完成，耗时: {enhance_time:.2f}秒")
        
        restore_start_time = time.time()
        # 将增强后的人脸还原到原始视频帧
        for index, face in enumerate(face_frames):
            try:
                out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
                out_frames.append(out_frame)
            except Exception as e:
                print(f"还原第 {index} 帧时出错: {str(e)}")
                out_frames.append(video_frames[index])
        
        restore_time = time.time() - restore_start_time
        total_time = time.time() - start_time
        
        print(f"\n人脸处理时间统计:")
        print(f"预处理时间: {preprocess_time:.2f}秒")
        print(f"人脸增强时间: {enhance_time:.2f}秒")
        print(f"还原时间: {restore_time:.2f}秒")
        print(f"总时间: {total_time:.2f}秒")
        print(f"平均每帧处理时间: {total_time/len(faces):.3f}秒")
        
        return np.stack(out_frames, axis=0)

    def loop_video(self, whisper_chunks: list, video_frames: np.ndarray):
        # If the audio is longer than the video, we need to loop the video
        if len(whisper_chunks) > len(video_frames):
            faces, boxes, affine_matrices, face_landmarks = self.affine_transform_video(video_frames)  # 修改：添加 face_landmarks
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_video_frames = []
            loop_faces = []
            loop_boxes = []
            loop_affine_matrices = []
            loop_face_landmarks = []  # 新增：存储循环的关键点
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_video_frames.append(video_frames)
                    loop_faces.append(faces)
                    loop_boxes += boxes
                    loop_affine_matrices += affine_matrices
                    loop_face_landmarks += face_landmarks  # 新增：添加关键点
                else:
                    loop_video_frames.append(video_frames[::-1])
                    loop_faces.append(faces.flip(0))
                    loop_boxes += boxes[::-1]
                    loop_affine_matrices += affine_matrices[::-1]
                    loop_face_landmarks += face_landmarks[::-1]  # 新增：添加反转的关键点

            video_frames = np.concatenate(loop_video_frames, axis=0)[: len(whisper_chunks)]
            faces = torch.cat(loop_faces, dim=0)[: len(whisper_chunks)]
            boxes = loop_boxes[: len(whisper_chunks)]
            affine_matrices = loop_affine_matrices[: len(whisper_chunks)]
            face_landmarks = loop_face_landmarks[: len(whisper_chunks)]  # 新增：截取关键点
        else:
            video_frames = video_frames[: len(whisper_chunks)]
            faces, boxes, affine_matrices, face_landmarks = self.affine_transform_video(video_frames)  # 修改：添加 face_landmarks

        return video_frames, faces, boxes, affine_matrices, face_landmarks  # 修改：返回关键点

    def init_face_enhancer(self, model_path=None, **kwargs):
        """初始化批处理人脸增强器
        
        Args:
            model_path: 模型路径
            **kwargs: 其他参数，包括 enhancement_method, enhancement_strength, 
                     mouth_protection, mouth_protection_strength 等
        """
        # 更新配置
        self.face_enhancer_config.update(kwargs)
        
        # 创建批处理增强器
        self.batch_face_enhancer = BatchFaceEnhancer(
            model_path=model_path,
            num_workers=self.face_enhancer_config.get('num_workers', 2),
            device=self.face_enhancer_config.get('device', 'cuda'),
            enhancement_method=self.face_enhancer_config.get('enhancement_method', 'gpen'),
            enhancement_strength=self.face_enhancer_config.get('enhancement_strength', 0.5),
            mouth_protection=self.face_enhancer_config.get('mouth_protection', True),
            mouth_protection_strength=self.face_enhancer_config.get('mouth_protection_strength', 0.8)
        )

    def enhance_video_frames(self, video_frames: np.ndarray, face_landmarks: list = None) -> np.ndarray:
        """批量增强视频帧
        
        Args:
            video_frames: 视频帧数组，形状为 [num_frames, height, width, channels]
            face_landmarks: 每帧的人脸关键点列表
            
        Returns:
            增强后的视频帧数组
        """
        if not self.face_enhancer_config['enable'] or self.batch_face_enhancer is None:
            return video_frames
            
        try:
            total_frames = len(video_frames)
            print(f"\n开始处理 {total_frames} 帧视频...")
            
            # 提交所有帧进行处理
            for i in range(total_frames):
                landmarks = face_landmarks[i] if face_landmarks is not None else None
                self.batch_face_enhancer.process_frame(i, video_frames[i], landmarks)
                if i % 10 == 0:
                    print(f"已提交 {i+1}/{total_frames} 帧")
            
            # 收集处理结果
            enhanced_frames = []
            processed_count = 0
            
            while processed_count < total_frames:
                try:
                    idx, frame = self.batch_face_enhancer.get_result(timeout=1.0)
                    if idx is not None and frame is not None:
                        enhanced_frames.append((idx, frame))
                        processed_count += 1
                        if processed_count % 10 == 0:
                            print(f"已处理 {processed_count}/{total_frames} 帧")
                except Exception as e:
                    print(f"获取结果时出错: {str(e)}")
                    continue
            
            print(f"处理完成，共处理 {processed_count} 帧")
            
            # 按帧索引排序结果
            enhanced_frames.sort(key=lambda x: x[0])
            return np.array([frame for _, frame in enhanced_frames])
            
        except Exception as e:
            print(f"批量增强视频帧时出错: {str(e)}")
            return video_frames

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        video_mask_path: str = None,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask: str = "fix_mask",
        mask_image_path: str = "latentsync/utils/mask.png",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        face_enhance: bool = False,
        face_enhance_method: str = 'gfpgan',
        face_enhance_strength: float = 0.5,
        mouth_protection: bool = True,
        mouth_protection_strength: float = 0.8,
        **kwargs,
    ):
        is_train = self.denoising_unet.training
        self.denoising_unet.eval()

        check_ffmpeg_installed()

        # 0. Define call parameters
        batch_size = 1
        device = self._execution_device
        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(height, mask=mask, device="cuda", mask_image=mask_image)
        self.image_processor.set_fps(video_fps)
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        # 1. Default height and width to unet
        height = height or self.denoising_unet.config.sample_size * self.vae_scale_factor
        width = width or self.denoising_unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

        audio_samples = read_audio(audio_path)
        video_frames = read_video(video_path, use_decord=False)

        video_frames, faces, boxes, affine_matrices, face_landmarks = self.loop_video(whisper_chunks, video_frames)  # 修改：添加 face_landmarks

        synced_video_frames = []
        masked_video_frames = []

        num_channels_latents = self.vae.config.latent_channels

        # Prepare latent variables
        all_latents = self.prepare_latents(
            batch_size,
            len(whisper_chunks),
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )

        num_inferences = math.ceil(len(whisper_chunks) / num_frames)
        for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
            if self.denoising_unet.add_audio_layer:
                audio_embeds = torch.stack(whisper_chunks[i * num_frames : (i + 1) * num_frames])
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
            else:
                audio_embeds = None
            inference_faces = faces[i * num_frames : (i + 1) * num_frames]
            latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]
            ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                inference_faces, affine_transform=False
            )

            # 7. Prepare mask latent variables
            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks,
                masked_pixel_values,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            # 8. Prepare image latents
            ref_latents = self.prepare_image_latents(
                ref_pixel_values,
                device,
                weight_dtype,
                generator,
                do_classifier_free_guidance,
            )

            # 9. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for j, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    denoising_unet_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    denoising_unet_input = self.scheduler.scale_model_input(denoising_unet_input, t)

                    # concat latents, mask, masked_image_latents in the channel dimension
                    denoising_unet_input = torch.cat(
                        [denoising_unet_input, mask_latents, masked_image_latents, ref_latents], dim=1
                    )

                    # predict the noise residual
                    noise_pred = self.denoising_unet(
                        denoising_unet_input, t, encoder_hidden_states=audio_embeds
                    ).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and j % callback_steps == 0:
                            callback(j, t, latents)

            # Recover the pixel values
            decoded_latents = self.decode_latents(latents)
            decoded_latents = self.paste_surrounding_pixels_back(
                decoded_latents, ref_pixel_values, 1 - masks, device, weight_dtype
            )
            synced_video_frames.append(decoded_latents)
            # masked_video_frames.append(masked_pixel_values)

        # 如果启用了人脸增强，初始化批处理增强器
        if face_enhance:
            self.init_face_enhancer(
                enhancement_method=face_enhance_method,
                enhancement_strength=face_enhance_strength,
                mouth_protection=mouth_protection,
                mouth_protection_strength=mouth_protection_strength
            )
        
        # # 在处理视频帧时使用批处理增强
        # if face_enhance and self.batch_face_enhancer is not None:
        #     video_frames = self.enhance_video_frames(video_frames, face_landmarks)

        synced_video_frames = self.restore_video(
            torch.cat(synced_video_frames), 
            video_frames, 
            boxes, 
            affine_matrices,
            face_landmarks
        )
        # masked_video_frames = self.restore_video(
        #     torch.cat(masked_video_frames), 
        #     video_frames, 
        #     boxes, 
        #     affine_matrices,
        #     face_landmarks
        # )

        audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.denoising_unet.train()

        temp_dir = "temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames, fps=25)
        # write_video(video_mask_path, masked_video_frames, fps=25)

        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)
