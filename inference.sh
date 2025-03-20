#!/bin/bash

# Default values
FACE_UPSCALE=1.0
HIGH_QUALITY=false

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
        *)
            # Save remaining arguments
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Build high quality flag
HQ_FLAG=""
if [ "$HIGH_QUALITY" = true ]; then
    HQ_FLAG="--high_quality"
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
  $HQ_FLAG \
  --seed 1247
