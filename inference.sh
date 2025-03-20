#!/bin/bash

# Default values
FACE_UPSCALE=1.0
HIGH_QUALITY=false
FACE_ENHANCE=false
FACE_ENHANCE_METHOD="combined"
FACE_ENHANCE_STRENGTH=0.8

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --face_upscale)
            FACE_UPSCALE="$2"
            shift 2
            ;;
        --high_quality)
            HIGH_QUALITY=true
            shift
            ;;
        --face_enhance)
            FACE_ENHANCE=true
            shift
            ;;
        --face_enhance_method)
            FACE_ENHANCE_METHOD="$2"
            shift 2
            ;;
        --face_enhance_strength)
            FACE_ENHANCE_STRENGTH="$2"
            shift 2
            ;;
        *)
            # Save remaining arguments
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Build flags
HQ_FLAG=""
if [ "$HIGH_QUALITY" = true ]; then
    HQ_FLAG="--high_quality"
fi

FE_FLAG=""
if [ "$FACE_ENHANCE" = true ]; then
    FE_FLAG="--face_enhance"
fi

# Run inference
python -m scripts.inference \
  --unet_config_path configs/unet/stage2.yaml \
  --inference_ckpt_path checkpoints/latentsync_unet.pt \
  --guidance_scale 1.5 \
  --video_path assets/demo1_video.mp4 \
  --audio_path assets/demo1_audio.wav \
  --video_out_path results/result.mp4 \
  --face_upscale_factor $FACE_UPSCALE \
  --face_enhance_method $FACE_ENHANCE_METHOD \
  --face_enhance_strength $FACE_ENHANCE_STRENGTH \
  $FE_FLAG \
  $HQ_FLAG \
  --seed 1247
