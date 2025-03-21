# 使用ONNX模型进行面部增强

MD-LatentSync现在支持使用ONNX格式模型进行面部增强，这降低了依赖复杂度并提高了推理速度。

## 准备ONNX模型

您需要下载或转换ONNX格式的面部增强模型。请将模型文件放在 `models/faceenhancer/` 目录下：

1. **GFPGAN**: `models/faceenhancer/GFPGANv1.4.onnx`
2. **CodeFormer**: `models/faceenhancer/codeformer.onnx`
3. **GPEN**: `models/faceenhancer/GPEN-BFR-512.onnx`

### 模型转换

如果您已有原始模型，可以使用以下方法将其转换为ONNX格式：

#### GFPGAN转换为ONNX

```python
import torch
from gfpgan import GFPGANer

# 初始化GFPGAN模型
model = GFPGANer(
    model_path='GFPGANv1.4.pth',
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

# 创建示例输入
dummy_input = torch.randn(1, 3, 512, 512).cuda()

# 导出ONNX模型
torch.onnx.export(
    model.gfpgan.net,  # 模型
    dummy_input,       # 示例输入
    "GFPGANv1.4.onnx", # 输出文件名
    opset_version=12,  # ONNX操作集版本
    input_names=["input"],  # 输入名称
    output_names=["output"], # 输出名称
    dynamic_axes={
        "input": {0: "batch_size"}, 
        "output": {0: "batch_size"}
    }
)
```

#### CodeFormer转换为ONNX

```python
import torch
from codeformer.basicsr.utils.registry import ARCH_REGISTRY

# 加载CodeFormer模型
model = ARCH_REGISTRY.get('CodeFormer')(
    dim_embd=512,
    codebook_size=1024,
    n_head=8,
    n_layers=9,
    connect_list=['32', '64', '128', '256']
).cuda()

# 加载预训练权重
checkpoint = torch.load('codeformer.pth')['params_ema']
model.load_state_dict(checkpoint)
model.eval()

# 创建示例输入
dummy_input = torch.randn(1, 3, 512, 512).cuda()

# 导出ONNX模型
torch.onnx.export(
    model,           # 模型
    dummy_input,     # 示例输入
    "codeformer.onnx", # 输出文件名
    opset_version=12,  # ONNX操作集版本
    input_names=["input"],  # 输入名称
    output_names=["output"], # 输出名称
    dynamic_axes={
        "input": {0: "batch_size"}, 
        "output": {0: "batch_size"}
    }
)
```

#### 使用ONNX Runtime Optimizer优化模型（可选）

安装onnxruntime-tools并优化模型以获得更好的性能：

```bash
pip install onnxruntime-tools

python -m onnxruntime_tools.optimizer_cli --input model.onnx --output model_optimized.onnx --optimization_level basic
```

## 使用说明

在MD-LatentSync中，通过以下参数控制面部增强：

```bash
python predict.py \
  --video_path input.mp4 \
  --audio_path input.wav \
  --face_enhance True \
  --face_enhance_method gfpgan \  # 选择'gfpgan', 'codeformer'或'gpen'
  --face_enhance_strength 0.8 \   # 增强强度，0.0-1.0
  --mouth_protection True \       # 是否保护嘴唇区域
  --mouth_protection_strength 0.8 # 嘴唇保护强度，0.0-1.0
```

也可以在gradio界面中使用相同的选项。

## 优势

使用ONNX模型进行面部增强有以下优势：

1. **依赖更少**：不再需要安装复杂的原始模型框架
2. **运行更快**：ONNX Runtime针对推理进行了优化
3. **部署更简单**：ONNX格式是跨平台标准，便于部署
4. **内存占用更小**：优化后的模型通常内存占用更低

## 疑难解答

如果遇到问题，请检查：

1. 确保ONNX模型文件存在且路径正确
2. 确保已安装onnxruntime-gpu（或CPU版本）
3. 对于CUDA执行，确保CUDA和cuDNN版本与onnxruntime-gpu兼容
4. 检查模型输入尺寸是否与预期相符

如果面部增强不起作用，系统会自动降级并返回未增强的图像。 