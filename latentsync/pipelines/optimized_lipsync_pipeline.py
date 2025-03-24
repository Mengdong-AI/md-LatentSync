import inspect
import math
import os
import shutil
from typing import Callable, List, Optional, Union
import subprocess
from concurrent.futures import ThreadPoolExecutor
import torch.multiprocessing as mp

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

logger = logging.get_logger(__name__)

class OptimizedLipsyncPipeline(DiffusionPipeline):
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
        
        # Initialize scheduler configuration
        self._init_scheduler(scheduler)
        
        # Register components
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
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    def _init_scheduler(self, scheduler):
        """Initialize scheduler with proper configuration"""
        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

    def parallel_process_inputs(self, video_path: str, audio_path: str, num_frames: int):
        """Process video and audio inputs in parallel"""
        
        def process_video():
            video_frames = read_video(video_path)
            # Preprocess video frames
            processed_frames = []
            for frame in video_frames:
                # Add your video frame preprocessing here
                processed_frame = frame / 255.0  # Normalize to [0,1]
                processed_frames.append(processed_frame)
            return np.array(processed_frames)
        
        def process_audio():
            audio_features = self.audio_encoder.extract_features(
                audio_path=audio_path,
                num_frames=num_frames
            )
            return audio_features
            
        # Submit tasks to thread pool
        future_video = self.executor.submit(process_video)
        future_audio = self.executor.submit(process_audio)
        
        # Wait for both tasks to complete
        video_frames = future_video.result()
        audio_features = future_audio.result()
        
        return video_frames, audio_features
        
    def batch_process_frames(self, frames, batch_size=4):
        """Process frames in batches"""
        num_frames = len(frames)
        processed_frames = []
        
        for i in range(0, num_frames, batch_size):
            batch = frames[i:i + batch_size]
            batch = torch.from_numpy(batch).to(self.device)
            
            # Process batch
            with torch.cuda.amp.autocast():
                processed_batch = self.process_batch(batch)
            
            processed_frames.extend(processed_batch.cpu().numpy())
            
            # Clear cache after each batch
            torch.cuda.empty_cache()
            
        return np.array(processed_frames)
        
    @torch.cuda.amp.autocast()
    def process_batch(self, batch):
        """Process a single batch of frames"""
        # Add your batch processing logic here
        return batch
        
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        num_frames: int = 25,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.0,
        batch_size: int = 4,
        **kwargs
    ):
        # Process inputs in parallel
        video_frames, audio_features = self.parallel_process_inputs(
            video_path, 
            audio_path,
            num_frames
        )
        
        # Process frames in batches
        processed_frames = self.batch_process_frames(
            video_frames,
            batch_size=batch_size
        )
        
        # Run inference
        with torch.cuda.amp.autocast():
            output = self.run_inference(
                processed_frames,
                audio_features,
                num_inference_steps,
                guidance_scale
            )
            
        # Save output
        write_video(video_out_path, output, fps=25)
        
        return output
        
    def run_inference(self, frames, audio_features, num_inference_steps, guidance_scale):
        """Run the actual inference with the processed inputs"""
        batch_size = len(frames)
        device = self.device
        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare latent variables
        latents = self.prepare_latents(
            batch_size=1,  # We process one video at a time
            num_frames=len(frames),
            num_channels_latents=4,  # VAE latent channels
            height=frames[0].shape[0],
            width=frames[0].shape[1],
            dtype=frames.dtype,
            device=device,
            generator=None
        )

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare extra kwargs for the scheduler step
        extra_step_kwargs = self.prepare_extra_step_kwargs(None, None)

        # Convert frames to latent space
        video_latents = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:min(i + batch_size, len(frames))]
            batch = torch.from_numpy(batch).to(device=device, dtype=self.vae.dtype)
            
            # Encode frames
            with torch.no_grad():
                batch_latents = self.vae.encode(batch).latent_dist.sample()
                batch_latents = (batch_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                video_latents.append(batch_latents)
        
        video_latents = torch.cat(video_latents, dim=0)
        video_latents = rearrange(video_latents, "f c h w -> 1 c f h w")

        # Prepare audio features
        audio_embeds = audio_features.to(device=device, dtype=self.denoising_unet.dtype)
        if do_classifier_free_guidance:
            audio_embeds = audio_embeds.repeat(2, 1, 1)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand latents for classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # Predict the noise residual
                with torch.cuda.amp.autocast():
                    noise_pred = self.denoising_unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=audio_embeds
                    )

                # Perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Compute previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # Update progress
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # Decode latents
        with torch.cuda.amp.autocast():
            video = self.decode_latents(latents)
            video = rearrange(video, "(b f) c h w -> b f h w c", f=len(frames))
            video = video.cpu().numpy()

        # Post-process video frames
        video = np.clip(video * 255, 0, 255).astype(np.uint8)
        return video[0]  # Return the first (and only) video 