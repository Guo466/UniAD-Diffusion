"""
JSONL mining 数据加载工具。

支持从 jsonl 文件加载指定样本用于训练/测试，实现「仅使用 jsonl 指定数据」的能力。
参考 main_diffusion_rl_v1_16_95 分支实现。
"""

import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple


def _parse_mining_entry(entry: str) -> Tuple[str, int, List[str]]:
    """
    Parse one mining entry string.
    Supported formats:
      - "<path>"
      - "<path> <times>"
      - "<path> <times> <tag1,tag2,...>"
      - "<path> <times> <tag1> <tag2> ..."
    """
    parts = str(entry).strip().split()
    if len(parts) == 0:
        return "", 1, []
    if len(parts) == 1:
        return parts[0], 1, []

    # Backward compatible: second token is repeat times when parseable
    try:
        times = int(parts[1])
        tail = parts[2:]
    except Exception:
        times = 1
        tail = parts[1:]

    tags: List[str] = []
    for token in tail:
        if "," in token:
            tags.extend([t.strip() for t in token.split(",") if t.strip()])
        elif token.strip():
            tags.append(token.strip())
    return parts[0], times, tags


def _parse_and_dump_chunk(args):
    """进程内解析 JSONL 行并写入分片文件。"""
    base_idx, lines, base_dir, tags = args
    for offset, line in enumerate(lines):
        item = json.loads(line)
        if tags:
            item["_mining_tags"] = tags
        path = os.path.join(base_dir, f"{base_idx + offset}.json")
        with open(path, "w") as fo:
            json.dump(item, fo, ensure_ascii=False)
    return len(lines)


def load_mining_overfit_shared(mining_files):
    """
    将 mining_file 拆分为独立 json 文件存入共享内存目录。
    - 共享目录: /dev/shm/mining_overfit/<md5>/
    - 所有 local_rank 参与计算，任务按 rank 分配
    返回: (shm_dir, length)
    """
    key_raw = "||".join(sorted(map(str, mining_files)))
    key = hashlib.md5(key_raw.encode()).hexdigest()
    base_dir = "/dev/shm/mining_overfit"
    os.makedirs(base_dir, exist_ok=True)
    shm_dir = os.path.join(base_dir, key)
    meta_path = os.path.join(shm_dir, "meta.json")

    try:
        import torch.distributed as dist  # type: ignore
        dist_inited = dist.is_available() and dist.is_initialized()
    except Exception:
        dist_inited = False
        dist = None  # type: ignore

    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except Exception:
        local_rank = 0

    try:
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    except Exception:
        local_world_size = 1

    def _load_and_split_distributed():
        """所有 local_rank 参与的分布式加载和写入。"""
        if local_rank == 0:
            os.makedirs(shm_dir, exist_ok=True)

        if dist_inited and dist is not None:
            try:
                dist.barrier()
            except Exception:
                pass

        total = 0
        if local_rank == 0:
            print(f"[Mining] Loading mining overfit data from {len(mining_files)} files with {local_world_size} ranks...")

        for mining_entry in mining_files:
            f_path, times, tags = _parse_mining_entry(mining_entry)
            if not f_path:
                continue
            if local_rank == 0:
                tag_info = f", tags={tags}" if tags else ""
                print(f"  - Reading: {f_path} (times={times}{tag_info})")

            if not os.path.exists(f_path):
                if local_rank == 0:
                    print(f"    [Warning] File not found: {f_path}, skipping...")
                continue

            if not f_path.endswith(".jsonl"):
                if local_rank == 0:
                    print(f"    [Error] Only .jsonl format is supported, got: {f_path}")
                raise SystemExit(1)

            with open(f_path, "r") as f:
                cnt = sum(1 for line in f if line.strip())

            if local_rank == 0:
                print(f"    Total {cnt} lines, use {times} times, distributing to {local_world_size} ranks...")

            if cnt == 0:
                continue

            start_idx = total
            total += cnt * times

            rank_chunk_size = math.ceil(cnt / local_world_size)
            rank_start = local_rank * rank_chunk_size
            rank_end = min(cnt, (local_rank + 1) * rank_chunk_size)

            if rank_start < rank_end:
                my_lines = []
                with open(f_path, "r") as f:
                    line_idx = 0
                    for line in f:
                        if not line.strip():
                            continue
                        if line_idx >= rank_end:
                            break
                        if line_idx >= rank_start:
                            for _ in range(times):
                                my_lines.append(line)
                        line_idx += 1
                my_cnt = len(my_lines)

                workers = min(8, my_cnt)
                chunk_size = math.ceil(my_cnt / workers)
                tasks = []
                for w in range(workers):
                    s = w * chunk_size
                    e = min(my_cnt, (w + 1) * chunk_size)
                    if s >= e:
                        break
                    global_idx = start_idx + rank_start * times + s
                    tasks.append((global_idx, my_lines[s:e], shm_dir, tags))

                if tasks:
                    with ProcessPoolExecutor(max_workers=workers) as tp:
                        list(tp.map(_parse_and_dump_chunk, tasks))

                del my_lines, tasks

                print(f"    [Rank {local_rank}] Dumped {my_cnt} items (lines {rank_start}-{rank_end}).")

        if dist_inited and dist is not None:
            try:
                dist.barrier()
            except Exception:
                pass

        if local_rank == 0:
            with open(meta_path, "w") as mf:
                json.dump({"length": total}, mf)
            print(f"[Mining] Total cached items: {total} -> {shm_dir}")

        return total

    if os.path.exists(meta_path):
        with open(meta_path, "r") as mf:
            meta = json.load(mf)
            return shm_dir, int(meta.get("length", 0))

    _ = _load_and_split_distributed()

    if dist_inited and dist is not None:
        try:
            dist.barrier()
        except Exception:
            pass

    if os.path.exists(meta_path):
        with open(meta_path, "r") as mf:
            meta = json.load(mf)
            return shm_dir, int(meta.get("length", 0))

    print(f"[Mining] meta.json missing in {shm_dir}, mining disabled.")
    return None, 0
