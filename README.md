<h1 align="center">LatentSync</h1>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b)](https://arxiv.org/abs/2412.09262)
[![arXiv](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/ByteDance/LatentSync-1.5)
[![arXiv](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/fffiloni/LatentSync)
<a href="https://replicate.com/lucataco/latentsync"><img src="https://replicate.com/lucataco/latentsync/badge" alt="Replicate"></a>

</div>

## 🔥 Updates

- `2025/03/14`: We released **LatentSync 1.5**, which **(1)** improves temporal consistency via adding temporal layer, **(2)** improves performance on Chinese videos and **(3)** reduces the VRAM requirement of the stage2 training to **20 GB** through a series of optimizations. Learn more details [here](docs/changelog_v1.5.md).

## 📖 Introduction

We present *LatentSync*, an end-to-end lip-sync method based on audio-conditioned latent diffusion models without any intermediate motion representation, diverging from previous diffusion-based lip-sync methods based on pixel-space diffusion or two-stage generation. Our framework can leverage the powerful capabilities of Stable Diffusion to directly model complex audio-visual correlations.

## 🏗️ Framework

<p align="center">
<img src="docs/framework.png" width=100%>
<p>

LatentSync uses the [Whisper](https://github.com/openai/whisper) to convert melspectrogram into audio embeddings, which are then integrated into the U-Net via cross-attention layers. The reference and masked frames are channel-wise concatenated with noised latents as the input of U-Net. In the training process, we use a one-step method to get estimated clean latents from predicted noises, which are then decoded to obtain the estimated clean frames. The TREPA, [LPIPS](https://arxiv.org/abs/1801.03924) and [SyncNet](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf) losses are added in the pixel space.

## 🎬 Demo

<table class="center">
  <tr style="font-weight: bolder;text-align:center;">
        <td width="50%"><b>Original video</b></td>
        <td width="50%"><b>Lip-synced video</b></td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/ff3a84da-dc9b-498a-950f-5c54f58dd5c5 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/150e00fd-381e-4421-a478-a9ea3d1212a8 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/32c830a9-4d7d-4044-9b33-b184d8e11010 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/84e4fe9d-b108-44a4-8712-13a012348145 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/7510a448-255a-44ee-b093-a1b98bd3961d controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/6150c453-c559-4ae0-bb00-c565f135ff41 controls preload></video>
    </td>
  </tr>
  <tr>
    <td width=300px>
      <video src=https://github.com/user-attachments/assets/0f7f9845-68b2-4165-bd08-c7bbe01a0e52 controls preload></video>
    </td>
    <td width=300px>
      <video src=https://github.com/user-attachments/assets/c34fe89d-0c09-4de3-8601-3d01229a69e3 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/7ce04d50-d39f-4154-932a-ec3a590a8f64 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/70bde520-42fa-4a0e-b66c-d3040ae5e065 controls preload></video>
    </td>
  </tr>
</table>

(Photorealistic videos are filmed by contracted models, and anime videos are from [VASA-1](https://www.microsoft.com/en-us/research/project/vasa-1/) and [EMO](https://humanaigc.github.io/emote-portrait-alive/))

## 📑 Open-source Plan

- [x] Inference code and checkpoints
- [x] Data processing pipeline
- [x] Training code

## 🔧 Setting up the Environment

Install the required packages and download the checkpoints via:

```bash
source setup_env.sh
```

If the download is successful, the checkpoints should appear as follows:

```
./checkpoints/
|-- latentsync_unet.pt
|-- stable_syncnet.pt
|-- whisper
|   `-- tiny.pt
|-- auxiliary
|   |-- 2DFAN4-cd938726ad.zip
|   |-- i3d_torchscript.pt
|   |-- koniq_pretrained.pkl
|   |-- s3fd-619a316812.pth
|   |-- sfd_face.pth
|   |-- syncnet_v2.model
|   |-- vgg16-397923af.pth
|   `-- vit_g_hybrid_pt_1200e_ssv2_ft.pth
```

These already include all the checkpoints required for latentsync training and inference. If you just want to try inference, you only need to download `latentsync_unet.pt` and `tiny.pt` from our [HuggingFace repo](https://huggingface.co/ByteDance/LatentSync-1.5)

## 🚀 Inference

There are two ways to perform inference, and both require **6.8 GB** of VRAM.

### 1. Gradio App

Run the Gradio app for inference:

```bash
python gradio_app.py
```

### 2. Command Line Interface

Run the script for inference:

```bash
./inference.sh
```

You can change the parameters `inference_steps` and `guidance_scale` to see more results.

## 🔄 Data Processing Pipeline

The complete data processing pipeline includes the following steps:

1. Remove the broken video files.
2. Resample the video FPS to 25, and resample the audio to 16000 Hz.
3. Scene detect via [PySceneDetect](https://github.com/Breakthrough/PySceneDetect).
4. Split each video into 5-10 second segments.
5. Affine transform the faces according to the landmarks detected by [face-alignment](https://github.com/1adrianb/face-alignment), then resize to 256 $\times$ 256.
6. Remove videos with [sync confidence score](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf) lower than 3, and adjust the audio-visual offset to 0.
7. Calculate [hyperIQA](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.pdf) score, and remove videos with scores lower than 40.

Run the script to execute the data processing pipeline:

```bash
./data_processing_pipeline.sh
```

You should change the parameter `input_dir` in the script to specify the data directory to be processed. The processed videos will be saved in the `high_visual_quality` directory. Each step will generate a new directory to prevent the need to redo the entire pipeline in case the process is interrupted by an unexpected error.

## 🏋️‍♂️ Training U-Net

Before training, you must process the data as described above and download all the checkpoints. We released a pretrained SyncNet with 94% accuracy on both VoxCeleb2 and HDTF datasets for the supervision of U-Net training. Note that this SyncNet is trained on affine transformed videos, so when using or evaluating this SyncNet, you need to perform affine transformation on the video first (the code of affine transformation is included in the data processing pipeline).

If all the preparations are complete, you can train the U-Net with the following script:

```bash
./train_unet.sh
```

We prepared three UNet configuration files in the ``configs/unet`` directory, each corresponding to a different training setup:

- `stage1.yaml`: Stage1 training, requires **23 GB** VRAM.
- `stage2.yaml`: Stage2 training with optimal performance, requires **30 GB** VRAM.
- `stage2_efficient.yaml`: Efficient Stage 2 training, requires **20 GB** VRAM. It may lead to slight degradation in visual quality and temporal consistency compared with `stage2.yaml`, suitable for users with consumer-grade GPUs, such as the RTX 3090.

Also remember to change the parameters in U-Net config file to specify the data directory, checkpoint save path, and other training hyperparameters.

## 🏋️‍♂️ Training SyncNet

In case you want to train SyncNet on your own datasets, you can run the following script. The data processing pipeline for SyncNet is the same as U-Net. 

```bash
./train_syncnet.sh
```

After `validations_steps` training, the loss charts will be saved in `train_output_dir`. They contain both the training and validation loss. If you want to customize the architecture of SyncNet for different image resolutions and input frame lengths, please follow the [guide](docs/syncnet_arch.md).

## 📊 Evaluation

You can evaluate the [sync confidence score](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf) of a generated video by running the following script:

```bash
./eval/eval_sync_conf.sh
```

You can evaluate the accuracy of SyncNet on a dataset by running the following script:

```bash
./eval/eval_syncnet_acc.sh
```

## 🙏 Acknowledgement

- Our code is built on [AnimateDiff](https://github.com/guoyww/AnimateDiff). 
- Some code are borrowed from [MuseTalk](https://github.com/TMElyralab/MuseTalk), [StyleSync](https://github.com/guanjz20/StyleSync), [SyncNet](https://github.com/joonson/syncnet_python), [Wav2Lip](https://github.com/Rudrabha/Wav2Lip).

Thanks for their generous contributions to the open-source community.

<!-- ## Citation
If you find our repo useful for your research, please consider citing our paper:
```

``` -->

# 面部增强功能

LatentSync现在支持三种高级面部增强方法，可以显著提高生成视频的质量：

## 可用的面部增强器

1. **GFPGAN**：腾讯ARC开发的高效面部恢复模型，能够有效修复低质量、模糊的面部图像。
2. **CodeFormer**：基于Transformer架构的面部恢复模型，提供可控的质量-保真度平衡。
3. **GPEN**：专注于保留原始面部特征的面部修复和增强模型。

## 安装

要使用面部增强功能，请按照以下步骤操作：

1. 运行安装脚本安装所需依赖：
   ```bash
   bash install_deps.sh
   ```

2. 脚本会询问是否下载面部增强模型，选择 'y' 以下载GFPGAN模型。
3. 根据需要，可以选择是否安装CodeFormer和GPEN。

## 使用方法

### 命令行使用

```bash
./inference.sh --face_enhance --face_enhance_method gfpgan --face_enhance_strength 0.8 --high_quality
```

面部增强参数说明：
- `--face_enhance`：启用面部增强
- `--face_enhance_method`：选择增强方法（gfpgan, codeformer, gpen）
- `--face_enhance_strength`：设置增强强度（0.0-1.0）

嘴唇保护参数说明：
- `--mouth_protection`：启用嘴唇保护（默认启用）
- `--no_mouth_protection`：禁用嘴唇保护
- `--mouth_protection_strength`：设置嘴唇保护强度（0.0-1.0），0表示完全保留原始嘴唇，1表示完全使用增强嘴唇

其他参数：
- `--high_quality`：使用高质量视频编码设置
- `--face_upscale_factor`：设置面部上采样因子（1.0-2.0）

### Gradio界面使用

1. 勾选 "Face Enhance" 选项
2. 从下拉菜单选择增强方法（GFPGAN, CodeFormer, GPEN）
3. 调整增强强度滑块
4. 勾选 "Mouth Protection" 选项以保护嘴唇区域
5. 调整嘴唇保护强度滑块
6. 点击处理按钮

## 嘴唇保护机制

该功能可以保护嘴唇区域，确保面部增强不会影响唇形同步效果：

1. **原理**：在应用面部增强过程中，识别并特殊处理嘴唇区域，以保持原始嘴型
2. **可控性**：用户可以调整保护强度，平衡嘴唇质量和同步效果
3. **默认设置**：默认启用嘴唇保护，强度0.8（部分保留原始嘴唇）

## 注意事项

1. 面部增强处理会增加处理时间，但能显著提高面部质量
2. CodeFormer和GPEN需要额外安装，请按照安装脚本提示完成安装
3. 在处理唇形同步视频时，建议保持嘴唇保护功能开启，以确保最佳同步效果
4. 如果结果视频的嘴唇质量过低，可以尝试调整嘴唇保护强度，在同步准确性和图像质量之间找到平衡

# MD-LatentSync

基于LatentSync的唇形同步系统，增加了高质量视频处理和面部增强功能。

## 新增功能

### 高质量视频处理
- 保留原始视频质量，只替换面部区域
- 使用高质量插值方法进行图像处理
- 支持可配置的面部放大系数
- 高质量视频编码，支持无损或低损失压缩

### 面部增强功能
- 集成多种面部增强方法:
  - GFPGAN: 默认方法，提供良好的面部修复和增强效果
  - CodeFormer: 适用于更细致的面部细节处理
  - GPEN: 提供替代的面部增强方案
  - 基础增强: 包括锐化、直方图均衡化和细节增强等
- 可调节增强强度
- 嘴唇区域保护功能，确保唇形同步效果

## 安装

1. 克隆仓库
```bash
git clone https://github.com/your-username/md-latentsync.git
cd md-latentsync
```

2. 安装依赖
```bash
bash install_deps_auto.sh
```
或者使用交互式安装:
```bash
bash install_deps.sh
```

## 使用方法

### 命令行使用
```bash
python inference.py \
  --audio_path path/to/audio.wav \
  --video_path path/to/video.mp4 \
  --face_upscale_factor 1.2 \
  --face_enhance True \
  --face_enhance_method gfpgan \
  --face_enhance_strength 0.8 \
  --mouth_protection True \
  --high_quality True
```

### 参数说明
- `--audio_path`: 音频文件路径
- `--video_path`: 视频文件路径
- `--face_upscale_factor`: 面部放大系数 (默认: 1.0)
- `--face_enhance`: 是否启用面部增强 (默认: False)
- `--face_enhance_method`: 面部增强方法，支持 'gfpgan', 'codeformer', 'gpen', 'sharpen', 'clahe', 'detail', 'combined' (默认: 'gfpgan')
- `--face_enhance_strength`: 面部增强强度，范围 0.0-1.0 (默认: 0.8)
- `--mouth_protection`: 是否保护嘴唇区域 (默认: True)
- `--mouth_protection_strength`: 嘴唇保护强度，范围 0.0-1.0 (默认: 0.8)
- `--high_quality`: 启用高质量视频输出 (默认: False)

### Gradio界面
使用交互式界面:
```bash
python gradio_app.py
```

## 高级配置
对于需要更精细控制的用户，可以编辑配置文件:
```bash
vi config/default.yaml
```

## 注意事项
- 面部增强模型需要额外的磁盘空间和内存
- 高质量处理可能增加处理时间
- 推荐使用CUDA加速

## 引用
如果您在研究中使用了此项目，请引用原始LatentSync项目:
```
@inproceedings{chu2023latentsync,
  title={LatentSync: Talking Face Generation with Precision using Synchronized Audio-Visual Latent Codes},
  author={Chu, Zhiming and Wang, Zhenyu and Zhang, Chuyi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```

## 高级特性

### 面部增强 (ONNX版本)

为简化依赖并提高性能，MD-LatentSync现在支持使用ONNX格式的面部增强模型。这需要将相应的ONNX模型文件放置在正确位置：

```
models/faceenhancer/GFPGANv1.4.onnx     # GFPGAN模型
models/faceenhancer/codeformer.onnx     # CodeFormer模型
models/faceenhancer/GPEN-BFR-512.onnx   # GPEN模型
```

详细说明和模型转换指南请参考 [docs/face_enhancer_onnx.md](docs/face_enhancer_onnx.md)。

要在运行时启用面部增强，使用以下参数：

```bash
python predict.py \
  --video_path input.mp4 \
  --audio_path input.wav \
  --face_enhance True \
  --face_enhance_method gfpgan \  # 'gfpgan', 'codeformer'或'gpen'
  --face_enhance_strength 0.8     # 增强强度 (0.0-1.0)
```

使用ONNX模型的好处：
- 依赖更少：不再需要安装原始模型框架
- 性能更佳：ONNX Runtime针对推理进行了优化
- 部署更简单：标准化的跨平台格式