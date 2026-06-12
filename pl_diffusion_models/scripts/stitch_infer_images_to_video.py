#!/usr/bin/env python3
"""
将推理输出目录下、各「日期时间」子文件夹内的图片按顺序拼成 MP4。

典型目录结构：
  <root>/infer/2026_04_08_22_59_37/path_compare/*.png

每个名称匹配 YYYY_MM_DD_HH_MM_SS 的文件夹生成一个视频，文件名为该文件夹名 + .mp4。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable, List

# 与现有输出目录命名一致：2026_04_08_22_59_37
DATE_DIR_RE = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def natural_key(s: str) -> List:
    """按路径分段、数字按数值排序，便于 2.png 在 10.png 之前。"""
    parts = re.split(r"(\d+)", s.lower())
    return [int(p) if p.isdigit() else p for p in parts]


def is_date_dir_name(name: str) -> bool:
    return bool(DATE_DIR_RE.match(name))


def iter_date_root_dirs(root: str) -> Iterable[str]:
    """遍历 root 下所有目录名符合日期时间模式的文件夹路径。"""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"不是有效目录: {root}")
    for dirpath, dirnames, _filenames in os.walk(root):
        base = os.path.basename(dirpath)
        if is_date_dir_name(base):
            yield dirpath


def collect_images_under(date_dir: str) -> List[str]:
    out: List[str] = []
    for dp, _dn, filenames in os.walk(date_dir):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTS:
                out.append(os.path.join(dp, f))
    out.sort(key=lambda p: natural_key(os.path.relpath(p, date_dir)))
    return out


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def write_concat_list(image_paths: List[str], fps: float, list_path: str) -> None:
    """ffmpeg concat demuxer：每张图显示 1/fps 秒；最后一行重复末帧以满足 demuxer 要求。"""
    if not image_paths:
        return
    duration = 1.0 / max(fps, 1e-6)
    lines: List[str] = []
    for p in image_paths:
        ap = os.path.abspath(p)
        # concat demuxer 中单引号需转义
        safe = ap.replace("'", "'\\''")
        lines.append(f"file '{safe}'")
        lines.append(f"duration {duration:.6f}")
    # 最后一帧再写一次 file，避免部分 ffmpeg 版本截断最后一帧时长
    ap = os.path.abspath(image_paths[-1])
    safe = ap.replace("'", "'\\''")
    lines.append(f"file '{safe}'")
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_ffmpeg_concat(list_file: str, out_mp4: str, crf: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_file,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        out_mp4,
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将 infer 输出根目录下各日期子文件夹内的图片拼成 MP4（以日期文件夹名命名）。"
    )
    parser.add_argument(
        "root",
        type=str,
        help="推理输出根目录，例如 /path/to/output/infer_navitopo_issues",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="视频输出目录，默认 <root>/videos",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="帧率（每秒切换张数），默认 5",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=20,
        help="libx264 CRF，默认 20（越小画质越好）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将处理的日期文件夹与图片数量，不调用 ffmpeg",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    out_dir = os.path.abspath(args.output_dir or os.path.join(root, "videos"))

    if not ffmpeg_available() and not args.dry_run:
        print("错误: 未找到 ffmpeg，请先安装或加入 PATH。", file=sys.stderr)
        return 1

    date_dirs = sorted(iter_date_root_dirs(root), key=lambda p: natural_key(p))
    if not date_dirs:
        print(f"在 {root} 下未发现名称形如 YYYY_MM_DD_HH_MM_SS 的子文件夹。", file=sys.stderr)
        return 1

    os.makedirs(out_dir, exist_ok=True)
    ok = 0
    skip = 0

    for date_dir in date_dirs:
        name = os.path.basename(date_dir)
        images = collect_images_under(date_dir)
        out_mp4 = os.path.join(out_dir, f"{name}.mp4")

        if not images:
            print(f"[跳过] {name}: 无图片")
            skip += 1
            continue

        if args.dry_run:
            print(f"[dry-run] {name}: {len(images)} 张 -> {out_mp4}")
            ok += 1
            continue

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tmp:
            tmp_path = tmp.name
        try:
            write_concat_list(images, args.fps, tmp_path)
            run_ffmpeg_concat(tmp_path, out_mp4, args.crf)
            print(f"[完成] {name}: {len(images)} 张 -> {out_mp4}")
            ok += 1
        except subprocess.CalledProcessError as e:
            print(f"[失败] {name}: ffmpeg 退出码 {e.returncode}", file=sys.stderr)
            skip += 1
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    print(f"结束: 成功 {ok}, 跳过/失败 {skip}, 输出目录 {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
