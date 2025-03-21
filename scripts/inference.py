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

import argparse
import os
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature


def main(config, args):
    try:
        if not os.path.exists(args.video_path):
            raise RuntimeError(f"Video path '{args.video_path}' not found")
        if not os.path.exists(args.audio_path):
            raise RuntimeError(f"Audio path '{args.audio_path}' not found")

        # Check if the GPU supports float16
        is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
        dtype = torch.float16 if is_fp16_supported else torch.float32

        print(f"Input video path: {args.video_path}")
        print(f"Input audio path: {args.audio_path}")
        print(f"Loaded checkpoint path: {args.inference_ckpt_path}")

        scheduler = DDIMScheduler.from_pretrained("configs")

        if config.model.cross_attention_dim == 768:
            whisper_model_path = "checkpoints/whisper/small.pt"
        elif config.model.cross_attention_dim == 384:
            whisper_model_path = "checkpoints/whisper/tiny.pt"
        else:
            raise NotImplementedError("cross_attention_dim must be 768 or 384")

        audio_encoder = Audio2Feature(
            model_path=whisper_model_path,
            device="cuda",
            num_frames=config.data.num_frames,
            audio_feat_length=config.data.audio_feat_length,
        )

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0

        denoising_unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            args.inference_ckpt_path,
            device="cpu",
        )

        denoising_unet = denoising_unet.to(dtype=dtype)

        pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            denoising_unet=denoising_unet,
            scheduler=scheduler,
        ).to("cuda")

        if args.seed != -1:
            set_seed(args.seed)
        else:
            torch.seed()

        print(f"Initial seed: {torch.initial_seed()}")
        print(f"Face upscale factor: {args.face_upscale_factor}")
        print(f"Face enhance: {'Enabled' if args.face_enhance else 'Disabled'}")
        print(f"Face enhance method: {args.face_enhance_method}")
        print(f"Face enhance strength: {args.face_enhance_strength}")
        print(f"Mouth protection: {'Enabled' if args.mouth_protection else 'Disabled'}")
        print(f"Mouth protection strength: {args.mouth_protection_strength}")
        print(f"High quality mode: {'Enabled' if args.high_quality else 'Disabled'}")

        pipeline(
            video_path=args.video_path,
            audio_path=args.audio_path,
            video_out_path=args.video_out_path,
            video_mask_path=args.video_out_path.replace(".mp4", "_mask.mp4"),
            num_frames=config.data.num_frames,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            face_upscale_factor=args.face_upscale_factor,
            face_enhance=args.face_enhance,
            face_enhance_method=args.face_enhance_method,
            face_enhance_strength=args.face_enhance_strength,
            mouth_protection=args.mouth_protection,
            mouth_protection_strength=args.mouth_protection_strength,
            high_quality=args.high_quality,
            weight_dtype=dtype,
            width=config.data.resolution,
            height=config.data.resolution,
            mask_image_path=config.data.mask_image_path,
        )
    except Exception as e:
        import traceback
        print(f"\n[ERROR] 处理过程中出现错误:")
        print(f"错误类型: {type(e)}")
        print(f"错误信息: {str(e)}")
        print(f"完整错误堆栈:\n{traceback.format_exc()}")
        raise  # 重新抛出异常，确保程序以非零状态退出


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--unet_config_path", type=str, required=True)
        parser.add_argument("--inference_ckpt_path", type=str, required=True)
        parser.add_argument("--video_path", type=str, required=True)
        parser.add_argument("--audio_path", type=str, required=True)
        parser.add_argument("--video_out_path", type=str, required=True)
        parser.add_argument("--inference_steps", type=int, default=20)
        parser.add_argument("--guidance_scale", type=float, default=1.5)
        parser.add_argument("--seed", type=int, default=-1)
        parser.add_argument("--face_upscale_factor", type=float, default=1.0)
        parser.add_argument("--face_enhance", action="store_true")
        parser.add_argument("--face_enhance_method", type=str, default="gfpgan")
        parser.add_argument("--face_enhance_strength", type=float, default=0.8)
        parser.add_argument("--mouth_protection", action="store_true")
        parser.add_argument("--mouth_protection_strength", type=float, default=0.8)
        parser.add_argument("--high_quality", action="store_true")
        args = parser.parse_args()

        config = OmegaConf.load(args.unet_config_path)
        main(config, args)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] 程序执行过程中出现错误:")
        print(f"错误类型: {type(e)}")
        print(f"错误信息: {str(e)}")
        print(f"完整错误堆栈:\n{traceback.format_exc()}")
        import sys
        sys.exit(1)  # 确保程序以非零状态退出
