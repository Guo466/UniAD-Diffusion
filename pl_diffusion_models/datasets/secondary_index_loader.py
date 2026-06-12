"""
二级索引分片加载工具。

这里的“二级索引”指按 scene 切分、以 `"{scene}_{shard_id}.json"` 命名的索引分片文件。

公开 API：
    - load_secondary_index_shards：按配置计算需要的二级索引分片，并并行加载到 /dev/shm，返回缓存目录
"""

import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, List

from .remote_client import create_remote_client

__all__ = [
    "load_secondary_index_shards",
    "get_secondary_index_tmp_dir",
    "build_shared_secondary_index_dict",
    "load_secondary_index_entry",
]


def _get_local_rank() -> int:
    """仅考虑 LOCAL_RANK；解析失败或缺失时默认 0。"""
    v = os.environ.get("LOCAL_RANK", "0")
    try:
        return int(v)
    except ValueError:
        return 0


def _get_local_world_size() -> int:
    """仅考虑 LOCAL_WORLD_SIZE；解析失败或缺失时默认 1。"""
    v = os.environ.get("LOCAL_WORLD_SIZE", "1")
    try:
        return int(v)
    except ValueError:
        return 1


def _get_shm_dir(remote_tmp_dir: str) -> str:
    """基于远端目录的 hash 生成 /dev/shm 下的唯一子目录。"""
    path_hash = hashlib.md5(remote_tmp_dir.encode()).hexdigest()
    return os.path.join("/dev/shm", "secondary_index", path_hash)


def load_secondary_index_shards(
    data_config,
    is_train: bool,
    enabled: bool = True,
    scene_extract_override=None,
) -> str:
    """
    加载需要用到的二级索引分片到 /dev/shm。

    返回：
        - 返回 /dev/shm 下的缓存目录路径
    """
    local_rank = _get_local_rank()
    rank_tag = f"local_rank={local_rank}"

    start_time = time.time()
    mode = "train" if is_train else "eval"
    print(f"[SecondaryIndex][{rank_tag}] start: load_secondary_index_shards (mode={mode})", flush=True)

    remote_tmp_dir = get_secondary_index_tmp_dir(data_config, is_train)

    if not enabled:
        elapsed_time = time.time() - start_time
        print(
            f"[SecondaryIndex][{rank_tag}] done: load_secondary_index_shards (mode={mode}, enabled=False), "
            f"elapsed={elapsed_time:.2f}s",
            flush=True,
        )
        return ""

    data_client = create_remote_client(data_config, cluster_path=None)

    scene_num, required_filenames = _collect_required_secondary_index_filenames(
        data_client=data_client,
        remote_tmp_dir=remote_tmp_dir,
        scene_extract_cfg=_resolve_scene_extract_cfg(data_config, is_train, scene_extract_override),
        max_frame=data_config.get("max_frame", float("inf")),
    )

    shm_dir = _get_shm_dir(remote_tmp_dir)
    os.makedirs(shm_dir, exist_ok=True)

    # 保存 scene_num.json
    shm_scene_num_path = os.path.join(shm_dir, "scene_num.json")
    if not os.path.exists(shm_scene_num_path) and local_rank == 0:
        with open(shm_scene_num_path, "w") as f:
            json.dump(scene_num, f)

    _download_secondary_index_files(
        data_config=data_config,
        remote_tmp_dir=remote_tmp_dir,
        required_filenames=required_filenames,
        shm_dir=shm_dir,
    )
    result_to_return = shm_dir

    elapsed_time = time.time() - start_time
    print(
        f"[SecondaryIndex][{rank_tag}] done: load_secondary_index_shards (mode={mode}), "
        f"elapsed={elapsed_time:.2f}s",
        flush=True,
    )

    return result_to_return


def build_shared_secondary_index_dict(
    data_config,
    is_train: bool,
    enabled: bool = True,
    scene_extract_override=None,
):
    """
    构建/填充二级索引共享存储（仅 /dev/shm），并在（可用时）进行分布式 barrier 同步。

    返回：
        - shm 目录路径字符串；禁用时返回空字符串
    """
    if not enabled:
        return ""

    result = load_secondary_index_shards(
        data_config=data_config,
        is_train=is_train,
        enabled=enabled,
        scene_extract_override=scene_extract_override,
    )

    # 可选：等待所有 rank 完成预取，避免训练阶段不同步
    try:
        import torch.distributed as dist  # type: ignore

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            if dist.get_rank() == 0:
                mode = "train" if is_train else "eval"
                print(f"[SecondaryIndex] all ranks ready (mode={mode})", flush=True)
    except Exception:
        # 没有 torch / 或 dist 不可用：直接跳过同步
        pass

    return result


def load_secondary_index_entry(
    scene: str,
    idx: int,
    prefetched_shm_dir: str,
    data_client,
    index_search_dirs: List[str],
    *,
    verbose: bool = False,
) -> Any:
    """
    加载单条二级索引分片（`{scene}_{idx//1000}.json`）并返回其 payload。

    读取顺序：
    1) 优先从 /dev/shm 目录 `prefetched_shm_dir` 获取
    2) 缺失则回退到 `index_search_dirs` 指向的目录（远端/本地）逐个查找并加载
    """
    filename = f"{scene}_{idx // 1000}.json"

    shm_path = os.path.join(prefetched_shm_dir, filename)
    if os.path.exists(shm_path):
        with open(shm_path, "r") as f:
            return json.load(f)

    if verbose:
        print(f"[SecondaryIndex] 警告: 索引文件 {filename} 不在共享内存中，回退到文件系统加载", flush=True)

    for base_dir in index_search_dirs:
        candidate = os.path.join(base_dir, filename)
        if data_client.contains(candidate):
            payload = data_client.load_json(candidate)
            if verbose:
                print(f"[SecondaryIndex] 已从文件系统加载: {candidate}", flush=True)
            return payload

    raise FileNotFoundError(f"{filename} not found in shm or index dirs: {index_search_dirs}")


def _get_secondary_index_tmp_dir(data_config, is_train: bool) -> str:
    # 历史上二级索引分片放在 tmp_data_dir（train）或 tmp_data_dir_test（eval）
    if is_train:
        return data_config["tmp_data_dir"]
    return data_config.get("tmp_data_dir_test", data_config["tmp_data_dir"])


def get_secondary_index_tmp_dir(data_config, is_train: bool) -> str:
    """返回 train/eval 对应的二级索引分片目录（历史上位于 tmp_data_dir / tmp_data_dir_test）。"""
    return _get_secondary_index_tmp_dir(data_config, is_train)


def _resolve_scene_extract_cfg(data_config, is_train: bool, override):
    if override is not None:
        return override
    if is_train:
        return data_config["scene_extract"]
    return data_config.get("scene_extract_test", data_config.get("scene_extract"))


def _collect_required_secondary_index_filenames(
    data_client,
    remote_tmp_dir: str,
    scene_extract_cfg,
    max_frame,
):
    """
    计算需要加载的二级索引分片文件名列表（如 scene_0.json）。
    分片粒度：每 1000 个“物理样本”对应 1 个 json 分片文件。
    """
    scene_num = None
    for candidate_dir in (
        remote_tmp_dir,
        os.path.dirname(remote_tmp_dir.rstrip('/')),
        os.path.join(os.path.dirname(remote_tmp_dir.rstrip('/')), "scene_statistics")
    ):
        candidate = os.path.join(candidate_dir, 'scene_num.json')
        if data_client.contains(candidate):
            scene_num = data_client.load_json(candidate)
            break
    if scene_num is None:
        raise FileNotFoundError(f"scene_num.json not found under {remote_tmp_dir} or its parent directory")

    try:
        max_frame = float(max_frame)
    except (TypeError, ValueError):
        max_frame = float("inf")

    required_files: List[str] = []
    consumed = 0
    for scene, extract_rate in scene_extract_cfg.items():
        if scene not in scene_num:
            continue

        total_frames = int(scene_num[scene] / extract_rate)
        remaining = float("inf") if math.isinf(max_frame) else max_frame - consumed
        if remaining <= 0:
            break

        use_frames = min(total_frames, remaining)
        # 计算实际需要访问的物理样本数（不是逻辑帧数）
        # 在 __getitem__ 中: idx_physical = idx_logical * extract_rate
        # 所以物理样本数 = 逻辑帧数 * extract_rate，且不超过场景总样本数
        physical_frames = min(scene_num[scene], math.ceil(use_frames * extract_rate))
        file_cnt = math.ceil(physical_frames / 1000)
        required_files.extend([f"{scene}_{i}.json" for i in range(file_cnt)])
        consumed += use_frames

    return scene_num, required_files


# 子进程内复用 RemoteClient，避免每个文件都重新创建 client/连接池
_MP_REMOTE_CLIENT = None
_MP_REMOTE_CLIENT_PID = None


def _load_secondary_index_partition_inflight_mp(args):
    """
    子进程内加载一整份分配到该进程的任务列表（partition），并在进程内用线程做 in-flight 并发。
    结果存储到 shm_dir。
    """
    tasks, data_config, threads_per_proc, shared_counter, shared_lock, shm_dir = args

    # 复用同一进程内的 RemoteClient（连接池）
    global _MP_REMOTE_CLIENT, _MP_REMOTE_CLIENT_PID
    pid = os.getpid()
    if _MP_REMOTE_CLIENT is None or _MP_REMOTE_CLIENT_PID != pid:
        _MP_REMOTE_CLIENT = create_remote_client(data_config, cluster_path=None)
        _MP_REMOTE_CLIENT_PID = pid

    if shm_dir:
        os.makedirs(shm_dir, exist_ok=True)

    errors: List[str] = []

    def _load_one(task):
        remote_path, filename = task

        # 如果是 shm 模式且文件已存在，跳过
        if shm_dir:
            shm_path = os.path.join(shm_dir, filename)
            if os.path.exists(shm_path):
                return filename, None

        try:
            payload = _MP_REMOTE_CLIENT.load_json(remote_path)

            # 存储到 shm
            if shm_dir:
                shm_path = os.path.join(shm_dir, filename)
                with open(shm_path, "w") as f:
                    json.dump(payload, f)

            return filename, None
        except Exception as e:
            return filename, e

    # 计数器更新：减少锁竞争，按批次累加
    local_completed = 0

    def _bump_counter(n: int):
        if shared_counter is None or shared_lock is None or n <= 0:
            return
        with shared_lock:
            shared_counter.value = int(shared_counter.value) + int(n)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    workers = threads_per_proc if (threads_per_proc > 1 and len(tasks) > 1) else 1
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as tp:
            futures = [tp.submit(_load_one, t) for t in tasks]
            for fut in as_completed(futures):
                filename, err = fut.result()
                if err is not None:
                    errors.append(f"{filename}: {err}")
                local_completed += 1
                if local_completed % 50 == 0:
                    _bump_counter(50)
    else:
        for t in tasks:
            filename, err = _load_one(t)
            if err is not None:
                errors.append(f"{filename}: {err}")
            local_completed += 1
            if local_completed % 50 == 0:
                _bump_counter(50)

    # flush remainder
    rem = local_completed % 50
    if rem:
        _bump_counter(rem)

    return local_completed, errors


def _download_secondary_index_files(
    data_config,
    remote_tmp_dir: str,
    required_filenames: List[str],
    shm_dir: str,
) -> None:
    """并行加载所需的二级索引分片，仅支持 shm 模式。"""
    local_rank = _get_local_rank()
    local_world_size = _get_local_world_size()
    is_rank0 = local_rank == 0

    os.makedirs(shm_dir, exist_ok=True)
    my_required_filenames = [
        f for i, f in enumerate(required_filenames) if i % local_world_size == local_rank
    ]
    if is_rank0:
        print(
            f"[SecondaryIndex] SHM Mode: total {len(required_filenames)} files, "
            f"rank {local_rank} handles {len(my_required_filenames)}",
            flush=True,
        )

    download_tasks = [(os.path.join(remote_tmp_dir, filename), filename) for filename in my_required_filenames]

    if not download_tasks:
        if is_rank0 and not required_filenames:
            print("[SecondaryIndex] no secondary index files to load")
        return

    mp_workers = int(os.environ.get("SECONDARY_INDEX_MULTIPROC_WORKERS", "8"))
    mp_workers = min(mp_workers, len(download_tasks))

    threads_per_proc = int(os.environ.get("SECONDARY_INDEX_THREADS_PER_PROC", "8"))
    threads_per_proc = max(1, threads_per_proc)

    partitions: List[List[tuple]] = [[] for _ in range(mp_workers)]
    for i, task in enumerate(download_tasks):
        partitions[i % mp_workers].append(task)

    start_time = time.time()
    completed = 0
    last_print_time = start_time

    pbar = None
    if is_rank0:
        try:
            import sys
            from tqdm import tqdm  # type: ignore

            lightning_like_bar_format = (
                "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            )
            pbar = tqdm(
                total=len(download_tasks),
                desc=f"SecondaryIndex(R{local_rank})",
                bar_format=lightning_like_bar_format,
                mininterval=0.5,
                dynamic_ncols=True,
                file=sys.stdout,
                leave=True,
            )
        except Exception:
            pbar = None

    error_count = 0
    sample_errors: List[str] = []

    shared_counter = None
    shared_lock = None
    if is_rank0 and pbar is not None:
        try:
            from multiprocessing import Manager
            mgr = Manager()
            shared_counter = mgr.Value("i", 0)
            shared_lock = mgr.Lock()
        except Exception:
            shared_counter = None
            shared_lock = None

    with ProcessPoolExecutor(max_workers=mp_workers) as executor:
        futures = [
            executor.submit(
                _load_secondary_index_partition_inflight_mp,
                (part, data_config, threads_per_proc, shared_counter, shared_lock, shm_dir),
            )
            for part in partitions
            if part
        ]

        last_counter_val = 0
        if is_rank0 and pbar is not None and shared_counter is not None and shared_lock is not None:
            while True:
                done_cnt = sum(1 for f in futures if f.done())
                with shared_lock:
                    cur = int(shared_counter.value)
                delta = cur - last_counter_val
                if delta > 0:
                    pbar.update(delta)
                    last_counter_val = cur
                if done_cnt == len(futures):
                    break
                time.sleep(0.5)

        for fut in as_completed(futures):
            part_completed, errors = fut.result()
            completed += part_completed
            error_count += len(errors)
            if errors and len(sample_errors) < 5:
                sample_errors.extend(errors[: (5 - len(sample_errors))])

            if pbar is None and is_rank0:
                now = time.time()
                # 这里的 completed 计数在 shm 模式下不太准，但不影响核心功能
                if completed % 2000 == 0 or completed >= len(download_tasks) or (now - last_print_time) >= 5.0:
                    elapsed = now - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(
                        f"[SecondaryIndex] rank {local_rank} progress: {completed}/{len(download_tasks)} files, {rate:.1f} files/s",
                        flush=True,
                    )
                    last_print_time = now

    if pbar is not None:
        if is_rank0 and shared_counter is not None and shared_lock is not None:
            with shared_lock:
                cur = int(shared_counter.value)
            if cur > last_counter_val:
                pbar.update(cur - last_counter_val)
        pbar.close()

    elapsed = time.time() - start_time
    completed = len(my_required_filenames)

    rate = completed / elapsed if elapsed > 0 else 0

    if error_count > 0:
        print(
            f"[SecondaryIndex][local_rank={local_rank}] completed with errors: "
            f"{completed}/{len(download_tasks)} files, elapsed={elapsed:.2f}s, "
            f"avg={rate:.1f} files/s, errors={error_count}",
            flush=True,
        )
        print(
            f"[SecondaryIndex][local_rank={local_rank}] sample_errors (up to 5): {sample_errors}",
            flush=True,
        )


