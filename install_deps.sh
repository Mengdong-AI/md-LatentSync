#!/bin/bash

# 创建用于存放模型的目录
mkdir -p models/faceenhancer

# 安装基本依赖
pip install -r requirements.txt

# 询问用户是否安装额外的面部增强模型
echo "是否安装额外的面部增强模型? (y/n)"
read -r INSTALL_ENHANCERS

if [[ "$INSTALL_ENHANCERS" == "y" ]]; then
    # 下载GFPGAN模型
    echo "正在下载GFPGAN模型..."
    wget -O models/faceenhancer/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

    # 询问是否安装CodeFormer
    echo "是否安装CodeFormer? (y/n)"
    read -r INSTALL_CODEFORMER
    
    if [[ "$INSTALL_CODEFORMER" == "y" ]]; then
        echo "正在安装CodeFormer..."
        git clone https://github.com/sczhou/CodeFormer.git
        cd CodeFormer
        pip install -r requirements.txt
        cd ..
        wget -O models/faceenhancer/codeformer.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
    fi
    
    # 询问是否安装GPEN
    echo "是否安装GPEN? (y/n)"
    read -r INSTALL_GPEN
    
    if [[ "$INSTALL_GPEN" == "y" ]]; then
        echo "正在安装GPEN..."
        git clone https://github.com/yangxy/GPEN.git
        cd GPEN
        pip install -r requirements.txt
        cd ..
        wget -O models/faceenhancer/GPEN-BFR-512.pth https://github.com/yangxy/GPEN/releases/download/v1.0.0/GPEN-BFR-512.pth
    fi
    
    echo "面部增强模型安装完成!"
else
    echo "跳过安装面部增强模型"
fi

echo "所有依赖安装完成!" 