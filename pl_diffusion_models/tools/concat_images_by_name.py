#!/usr/bin/env python3
"""
将 2 个文件夹下的 PNG 按文件名匹配后横向拼接，保存为 {原名}_concat.png 到指定输出目录。
用法示例:
  python tools/concat_images_by_name.py --dir1 path/to/folder1 --dir2 path/to/folder2 --out path/to/output
"""
import argparse
import os
import os.path as osp

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="按文件名匹配 2 个文件夹的 PNG，横向拼接后保存")
    parser.add_argument("--dir1", type=str, required=True, help="第 1 个图片文件夹")
    parser.add_argument("--dir2", type=str, required=True, help="第 2 个图片文件夹")
    parser.add_argument("--out", type=str, required=True, help="拼接结果保存目录")
    parser.add_argument("--suffix", type=str, default="_concat", help="输出文件名后缀，默认 _concat，最终为 {name}{suffix}.png")
    args = parser.parse_args()

    dir1 = osp.abspath(args.dir1)
    dir2 = osp.abspath(args.dir2)
    out_dir = osp.abspath(args.out)

    for d in [dir1, dir2]:
        if not osp.isdir(d):
            raise FileNotFoundError(f"目录不存在: {d}")
    os.makedirs(out_dir, exist_ok=True)

    def png_basenames(d):
        names = set()
        for f in os.listdir(d):
            if f.lower().endswith(".png"):
                names.add(osp.splitext(f)[0])
        return names

    set1 = png_basenames(dir1)
    set2 = png_basenames(dir2)
    common = set1 & set2
    if not common:
        print("未找到在两个文件夹中均存在的同名 PNG 文件名，退出")
        return

    print(f"共找到 {len(common)} 组可拼接的图片，开始拼接...")
    for name in sorted(common):
        path1 = osp.join(dir1, name + ".png")
        path2 = osp.join(dir2, name + ".png")
        im1 = cv2.imread(path1)
        im2 = cv2.imread(path2)
        if im1 is None or im2 is None:
            print(f"  跳过 {name}: 某张图读取失败")
            continue
        # 若高度不一致，按最大高度对齐并居中
        h1, h2 = im1.shape[0], im2.shape[0]
        h_max = max(h1, h2)
        if h1 < h_max or h2 < h_max:
            def pad_to_height(img, target_h):
                h, w = img.shape[:2]
                if h >= target_h:
                    return img
                pad_top = (target_h - h) // 2
                pad_bot = target_h - h - pad_top
                return np.pad(img, ((pad_top, pad_bot), (0, 0), (0, 0)), mode="constant", constant_values=255)
            im1 = pad_to_height(im1, h_max)
            im2 = pad_to_height(im2, h_max)
        concat = np.concatenate([im1, im2], axis=1)
        out_name = name + args.suffix + ".png"
        out_path = osp.join(out_dir, out_name)
        cv2.imwrite(out_path, concat)
        print(f"  已保存: {out_name}")
    print("完成")


if __name__ == "__main__":
    main()
