#!/bin/bash

# 创建安装不依赖交互的安装脚本
cat > install_deps_auto.sh << 'EOF'
#!/bin/bash

# 设置变量
INSTALL_ENHANCERS=true  # 是否安装面部增强器模型
INSTALL_CODEFORMER=false  # 是否安装CodeFormer
INSTALL_GPEN=false  # 是否安装GPEN

# 创建用于存放模型的目录
mkdir -p models/faceenhancer

# 安装基本依赖
echo "正在安装基本依赖..."
pip install -r requirements.txt

# 检查GFPGAN安装是否成功
if python -c "import gfpgan" 2>/dev/null; then
    echo "GFPGAN安装成功"
else
    echo "GFPGAN安装失败，尝试手动安装..."
    pip install git+https://github.com/TencentARC/GFPGAN.git --no-deps
    pip install basicsr facexlib
fi

if [ "$INSTALL_ENHANCERS" = true ]; then
    # 下载GFPGAN模型
    echo "正在下载GFPGAN模型..."
    mkdir -p models/faceenhancer
    if [ ! -f "models/faceenhancer/GFPGANv1.4.pth" ]; then
        wget -O models/faceenhancer/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
    else
        echo "GFPGAN模型已存在，跳过下载"
    fi

    if [ "$INSTALL_CODEFORMER" = true ]; then
        echo "正在安装CodeFormer..."
        if [ ! -d "CodeFormer" ]; then
            git clone https://github.com/sczhou/CodeFormer.git
            cd CodeFormer
            pip install -r requirements.txt --no-deps
            pip install scipy lpips einops timm
            cd ..
        else
            echo "CodeFormer已存在，跳过克隆"
        fi
        
        if [ ! -f "models/faceenhancer/codeformer.pth" ]; then
            wget -O models/faceenhancer/codeformer.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
        else
            echo "CodeFormer模型已存在，跳过下载"
        fi
    fi
    
    if [ "$INSTALL_GPEN" = true ]; then
        echo "正在安装GPEN..."
        if [ ! -d "GPEN" ]; then
            git clone https://github.com/yangxy/GPEN.git
            cd GPEN
            pip install -r requirements.txt --no-deps
            cd ..
        else
            echo "GPEN已存在，跳过克隆"
        fi
        
        if [ ! -f "models/faceenhancer/GPEN-BFR-512.pth" ]; then
            wget -O models/faceenhancer/GPEN-BFR-512.pth https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-512.pth
        else
            echo "GPEN模型已存在，跳过下载"
        fi
    fi
    
    echo "面部增强模型安装完成!"
else
    echo "跳过安装面部增强模型"
fi

echo "所有依赖安装完成!"
EOF

chmod +x install_deps_auto.sh
echo "已创建自动安装脚本 install_deps_auto.sh，您可以编辑该脚本自定义安装选项，然后执行它完成安装。" 