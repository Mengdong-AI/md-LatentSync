# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import math
import os
import shutil
from typing import Callable, List, Optional, Union
import subprocess

import numpy as np
import torch
import torchvision

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
from decord import VideoReader

from ..models.unet import UNet3DConditionModel
from ..utils.util import read_video, read_audio, write_video, check_ffmpeg_installed
from ..utils.image_processor import ImageProcessor, load_fixed_mask
from ..whisper.audio2feature import Audio2Feature
from ..utils.face_enhancer import FaceEnhancer  # 导入面部增强器
import tqdm
import soundfile as sf

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
        print(f"Affine transforming {len(video_frames)} faces...")
        for frame in tqdm.tqdm(video_frames):
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)

        faces = torch.stack(faces)
        return faces, boxes, affine_matrices

    def restore_video(self, faces, boxes, affine_matrices, source_video_path, opt_face_enhancer=None, original_aspect_ratio=True):
        """Restore video with aligned faces.
        
        Args:
            faces (np.ndarray): Aligned faces, with shape [T, 1, 3, H, W].
            boxes (np.ndarray): Face boxes, with shape [T, 1, 4].
            affine_matrices (np.ndarray): Affine matrices, with shape [T, 1, 2, 3].
            source_video_path (str): Path to original video.
            opt_face_enhancer (FaceEnhancer, optional): Face enhancer. Defaults to None.
            original_aspect_ratio (bool, optional): Whether to keep original aspect ratio. Defaults to True.
            
        Returns:
            np.ndarray: Restored video, with shape [T, H, W, 3].
        """
        # Create debug directory if not exists
        debug_dir = os.path.join(os.path.dirname(source_video_path), "debug_frames")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Read original video
        vr = VideoReader(source_video_path)
        # Get total frame count
        total_frames = len(vr)
        
        # Get original shapes
        original_frames = vr[:]  # [T, H, W, C]
        
        # 检查 original_frames 是否为可索引对象
        if not hasattr(original_frames, "__getitem__"):
            print(f"Warning: Cannot index original_frames, converting to numpy array")
            try:
                original_frames = original_frames.asnumpy() if hasattr(original_frames, "asnumpy") else np.array(original_frames)
            except:
                print(f"Error: Failed to convert original_frames to indexable object")
                # 如果转换失败，创建一个空帧列表
                original_frames = []
        
        # 如果 original_frames 为空或无法索引，使用 faces 创建一个空背景
        if len(original_frames) == 0:
            print(f"Warning: No original frames loaded, creating blank background")
            # 从第一个 face 获取尺寸信息
            face_h, face_w = faces[0, 0].shape[1:3] 
            # 创建一个黑色背景帧列表
            original_frames = [np.zeros((face_h*2, face_w*2, 3), dtype=np.uint8) for _ in range(min(len(faces), total_frames))]
        
        # 获取原始视频的高度和宽度
        if len(original_frames) > 0:
            if isinstance(original_frames, list):
                original_h, original_w = original_frames[0].shape[:2]
            else:
                original_h, original_w = original_frames.shape[1:3]
        else:
            # 如果没有原始帧，使用面部尺寸的2倍作为原始尺寸
            face_h, face_w = faces[0, 0].shape[1:3]
            original_h, original_w = face_h*2, face_w*2
        
        # Initialize output frames
        output_frames = []
        
        # Iterate through frames
        for i in range(min(len(faces), total_frames)):
            try:
                # Get current frame
                if len(original_frames) > 0:
                    if isinstance(original_frames, list):
                        ori_frame = original_frames[min(i, len(original_frames)-1)].copy()
                    else:
                        ori_frame = original_frames[min(i, len(original_frames)-1)].copy() if hasattr(original_frames, "copy") else np.array(original_frames[min(i, len(original_frames)-1)])
                else:
                    # 如果没有原始帧，创建黑色背景
                    ori_frame = np.zeros((original_h, original_w, 3), dtype=np.uint8)
                
                # Save original frame for first 5 frames
                if i < 5:
                    cv2.imwrite(os.path.join(debug_dir, f"frame_{i:03d}_1_original.png"), cv2.cvtColor(ori_frame, cv2.COLOR_RGB2BGR))
                
                # Convert to BGR for OpenCV processing
                ori_frame_bgr = cv2.cvtColor(ori_frame, cv2.COLOR_RGB2BGR)
                
                # Process face
                face = faces[i, 0]  # [3, H, W]
                face = rearrange(face, 'c h w -> h w c')  # [H, W, C]
                box = boxes[i, 0]  # [4,]
                affine_matrix = affine_matrices[i, 0]  # [2, 3]
                
                if face.shape[0] == 0 or box.sum() == 0:
                    # Skip invalid face
                    output_frames.append(ori_frame)
                    continue
                
                # Convert face from RGB to BGR (for OpenCV)
                face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                
                # Save aligned face for first 5 frames
                if i < 5:
                    cv2.imwrite(os.path.join(debug_dir, f"frame_{i:03d}_2_aligned_face.png"), face_bgr)
                
                # Resize face to 512x512 for enhancement if needed
                if opt_face_enhancer is not None and opt_face_enhancer.enable:
                    face_enhanced = cv2.resize(face_bgr, (512, 512), interpolation=cv2.INTER_LANCZOS4)
                    face_enhanced = opt_face_enhancer.enhance(face_enhanced)
                    face_bgr = cv2.resize(face_enhanced, (face_bgr.shape[1], face_bgr.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                    
                    # Save enhanced face for first 5 frames
                    if i < 5:
                        cv2.imwrite(os.path.join(debug_dir, f"frame_{i:03d}_3_enhanced_face.png"), face_bgr)
                
                # Compute inverse affine matrix
                inv_affine_matrix = cv2.invertAffineTransform(affine_matrix)
                
                # Get frame size to warp back
                frame_h, frame_w = ori_frame_bgr.shape[:2]
                
                # Create mask for face
                mask = np.ones((face_bgr.shape[0], face_bgr.shape[1], 1), dtype=np.float32)
                mask = cv2.warpAffine(mask, inv_affine_matrix, (frame_w, frame_h))
                mask = cv2.GaussianBlur(mask, (31, 31), 10)
                
                # Warp face back to original position
                warped_face = cv2.warpAffine(face_bgr, inv_affine_matrix, (frame_w, frame_h))
                
                # Save warped face and mask for first 5 frames
                if i < 5:
                    cv2.imwrite(os.path.join(debug_dir, f"frame_{i:03d}_4_warped_face.png"), warped_face)
                    cv2.imwrite(os.path.join(debug_dir, f"frame_{i:03d}_4_mask.png"), (mask * 255).astype(np.uint8))
                
                # Blend face with original frame
                warped_face = warped_face.astype(np.float32) / 255.0
                ori_frame_bgr = ori_frame_bgr.astype(np.float32) / 255.0
                
                output_frame = ori_frame_bgr * (1 - mask) + warped_face * mask
                output_frame = (output_frame * 255.0).astype(np.uint8)
                
                # Convert back to RGB
                output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                
                # Save final result for first 5 frames
                if i < 5:
                    cv2.imwrite(os.path.join(debug_dir, f"frame_{i:03d}_5_final.png"), cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
                
                output_frames.append(output_frame)
            
            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                # If error, try to use original frame or create a blank one
                try:
                    if len(original_frames) > 0 and i < len(original_frames):
                        if isinstance(original_frames, list):
                            output_frames.append(original_frames[i])
                        else:
                            output_frames.append(np.array(original_frames[i]))
                    else:
                        # 创建空白帧
                        blank_frame = np.zeros((original_h, original_w, 3), dtype=np.uint8)
                        output_frames.append(blank_frame)
                except Exception as e2:
                    print(f"Error creating fallback frame: {e2}")
                    # 最后的备用方案：创建一个小的空白帧
                    output_frames.append(np.zeros((256, 256, 3), dtype=np.uint8))
        
        # 确保至少有一帧
        if len(output_frames) == 0:
            print("Warning: No frames processed, creating a blank frame")
            output_frames = [np.zeros((original_h, original_w, 3), dtype=np.uint8)]
        
        # Stack frames
        output_frames = np.stack(output_frames, axis=0)
        
        print(f"Debug frames saved to: {debug_dir}")
        return output_frames

    def loop_video(self, whisper_chunks: list, video_frames: np.ndarray):
        # If the audio is longer than the video, we need to loop the video
        if len(whisper_chunks) > len(video_frames):
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_video_frames = []
            loop_faces = []
            loop_boxes = []
            loop_affine_matrices = []
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_video_frames.append(video_frames)
                    loop_faces.append(faces)
                    loop_boxes += boxes
                    loop_affine_matrices += affine_matrices
                else:
                    loop_video_frames.append(video_frames[::-1])
                    loop_faces.append(faces.flip(0))
                    loop_boxes += boxes[::-1]
                    loop_affine_matrices += affine_matrices[::-1]

            video_frames = np.concatenate(loop_video_frames, axis=0)[: len(whisper_chunks)]
            faces = torch.cat(loop_faces, dim=0)[: len(whisper_chunks)]
            boxes = loop_boxes[: len(whisper_chunks)]
            affine_matrices = loop_affine_matrices[: len(whisper_chunks)]
        else:
            video_frames = video_frames[: len(whisper_chunks)]
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)

        return video_frames, faces, boxes, affine_matrices

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
        face_upscale_factor: float = 1.0,  # 控制面部放大因子
        face_enhance: bool = False,  # 是否启用面部增强
        face_enhance_method: str = 'gfpgan',  # 面部增强方法
        face_enhance_strength: float = 0.8,  # 面部增强强度
        mouth_protection: bool = True,  # 是否保护嘴唇区域
        mouth_protection_strength: float = 0.8,  # 嘴唇保护强度
        high_quality: bool = False,  # 控制视频质量
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask: str = "fix_mask",
        mask_image_path: str = "latentsync/utils/mask.png",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        is_train = self.denoising_unet.training
        self.denoising_unet.eval()

        check_ffmpeg_installed()

        # 0. Define call parameters
        batch_size = 1
        device = self._execution_device
        mask_image = load_fixed_mask(height, mask_image_path)
        # 设置面部放大因子
        self.image_processor = ImageProcessor(height, mask=mask, device="cuda", mask_image=mask_image)
        self.image_processor.restorer.upscale_factor = face_upscale_factor
        
        # 设置面部增强器
        self.face_enhancer = None
        if face_enhance:
            self.face_enhancer = FaceEnhancer(
                enhancement_strength=face_enhance_strength,
                method=face_enhance_method,
                mouth_protection=mouth_protection,
                mouth_protection_strength=mouth_protection_strength
            )
            print(f"面部增强已启用 - 方法: {face_enhance_method}, 强度: {face_enhance_strength}")
            print(f"嘴唇保护: {'已启用' if mouth_protection else '已禁用'}, 保护强度: {mouth_protection_strength}")
            
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

        video_frames, faces, boxes, affine_matrices = self.loop_video(whisper_chunks, video_frames)

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

        # 将PyTorch张量转换为numpy数组以匹配新的restore_video参数格式
        faces_array = torch.cat(synced_video_frames).cpu().numpy()
        faces_array = np.expand_dims(faces_array, axis=1)  # [T, 1, C, H, W]
        
        boxes_array = np.array(boxes)
        boxes_array = np.expand_dims(boxes_array, axis=1)  # [T, 1, 4]
        
        affine_matrices_array = np.array(affine_matrices)
        affine_matrices_array = np.expand_dims(affine_matrices_array, axis=1)  # [T, 1, 2, 3]
        
        # 使用新的restore_video方法
        synced_video_frames = self.restore_video(
            faces=faces_array,
            boxes=boxes_array,
            affine_matrices=affine_matrices_array,
            source_video_path=video_path,
            opt_face_enhancer=self.face_enhancer,
            original_aspect_ratio=True
        )
        # masked_video_frames = self.restore_video(
        #     torch.cat(masked_video_frames), video_frames, boxes, video_path, self.face_enhancer
        # )

        audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.denoising_unet.train()

        temp_dir = "temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        # 根据是否开启高质量选项使用不同的视频写入方式
        if high_quality:
            # 使用ffmpeg直接将帧写入视频，而不是OpenCV
            temp_video_path = os.path.join(temp_dir, "video.mp4")
            temp_frames_dir = os.path.join(temp_dir, "frames")
            os.makedirs(temp_frames_dir, exist_ok=True)
            
            print(f"Saving {len(synced_video_frames)} frames...")
            # 先保存所有帧为图像文件
            for i, frame in enumerate(synced_video_frames):
                # synced_video_frames 现在是RGB格式，需要转换为BGR给OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_path = os.path.join(temp_frames_dir, f"frame_{i:04d}.png")
                cv2.imwrite(frame_path, frame_bgr)
            
            # 写入音频
            audio_path = os.path.join(temp_dir, "audio.wav")
            sf.write(audio_path, audio_samples, audio_sample_rate)
            
            # 使用ffmpeg将图像序列转换为视频
            imgs_path = os.path.join(temp_frames_dir, "frame_%04d.png")
            temp_video_no_audio = os.path.join(temp_dir, "video_no_audio.mp4")
            
            # 先创建没有音频的视频
            ffmpeg_cmd = (f"ffmpeg -y -loglevel error -framerate {video_fps} -i {imgs_path} "
                          f"-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p {temp_video_no_audio}")
            print("Running ffmpeg to create video...")
            subprocess.run(ffmpeg_cmd, shell=True)
            
            # 添加音频到视频
            ffmpeg_audio_cmd = (f"ffmpeg -y -loglevel error -i {temp_video_no_audio} "
                              f"-i {audio_path} -c:v copy -c:a aac -b:a 320k -shortest {video_out_path}")
            print("Adding audio to video...")
            subprocess.run(ffmpeg_audio_cmd, shell=True)
        else:
            # 使用原来的方式处理，写入视频函数会自动处理RGB->BGR转换
            write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames, fps=video_fps)
        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)
