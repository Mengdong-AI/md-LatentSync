#!/bin/bash

# 设置颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

# 创建目录
echo -e "${GREEN}创建必要目录...${NC}"
mkdir -p models/faceenhancer
mkdir -p output

# 检查系统
echo -e "${GREEN}检查系统环境...${NC}"
OS=$(uname -s)
if [ "$OS" = "Linux" ]; then
    echo "检测到Linux系统"
elif [ "$OS" = "Darwin" ]; then
    echo "检测到MacOS系统"
else
    echo -e "${YELLOW}无法确定操作系统类型，假设为Windows${NC}"
    OS="Windows"
fi

# 检查CUDA
echo -e "${GREEN}检查CUDA环境...${NC}"
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "检测到CUDA版本: $CUDA_VERSION"
    HAS_CUDA=1
elif [ -d "/usr/local/cuda" ]; then
    echo "检测到CUDA目录，但无法确定版本"
    HAS_CUDA=1
else
    echo -e "${YELLOW}未检测到CUDA，将安装CPU版本${NC}"
    HAS_CUDA=0
fi

# 安装基础依赖
echo -e "${GREEN}安装基础依赖...${NC}"
pip install -r requirements.txt

# 安装ONNX运行时
echo -e "${GREEN}安装ONNX Runtime...${NC}"
if [ $HAS_CUDA -eq 1 ]; then
    echo "安装ONNX Runtime GPU版本..."
    pip install onnxruntime-gpu
else
    echo "安装ONNX Runtime CPU版本..."
    pip install onnxruntime
fi

# 检查并创建模型目录
echo -e "${GREEN}检查模型目录...${NC}"
MODEL_DIRS=(
    "models/faceenhancer"
)

for DIR in "${MODEL_DIRS[@]}"; do
    if [ ! -d "$DIR" ]; then
        echo "创建目录: $DIR"
        mkdir -p "$DIR"
    fi
done

# 添加ONNX模型说明
echo -e "${YELLOW}请将以下ONNX模型文件放置在对应目录:${NC}"
echo "- models/faceenhancer/GFPGANv1.4.onnx (GFPGAN模型)"
echo "- models/faceenhancer/codeformer.onnx (CodeFormer模型)"
echo "- models/faceenhancer/GPEN-BFR-512.onnx (GPEN模型)"
echo -e "${YELLOW}如需了解更多信息，请参考 docs/face_enhancer_onnx.md${NC}"

# 最后的设置检查
echo -e "${GREEN}依赖安装完成，进行最终检查...${NC}"

# 检查torch是否正确安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"

# 检查onnxruntime是否正确安装
python -c "import onnxruntime; print(f'ONNX Runtime版本: {onnxruntime.__version__}')"
python -c "import onnxruntime; print(f'可用执行提供程序: {onnxruntime.get_available_providers()}')"

echo -e "${GREEN}安装完成!${NC}"
echo "如有任何问题，请查看文档或提交Issue。" 