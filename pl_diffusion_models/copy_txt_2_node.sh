#!/usr/bin/env bash

# 切换到目标目录
cd tools/ || {
    echo "❌ 切换目录失败"
    exit 1
}

echo "✅ 当前目录: $(pwd)"

# 创建目标目录
mkdir -p to_sdk || {
    echo "❌ 创建目录失败"
    exit 1
}

# 复制文本文件
cp -v ./deploy/*.txt to_sdk/ || {
    echo "⚠️ 复制文本文件时出错"
}

# 复制参数文件
cp -v ./deploy/parameters.json to_sdk/ || {
    echo "❌ 复制 parameters.json 失败"
    exit 1
}

scp to_sdk/* panwenbo@10.151.179.38:/home/SENSETIME/panwenbo/ws/pilotmdc_aarch64/dlp_cnop/senseauto-perception-camera/node/resource/models/perception_camera/thor/vd_dlp/dlp_v1_009_a_occ/

echo "🎉 操作完成！"

