"""
二级索引分片加载工具。

这里的“二级索引”指按 scene 切分、以 `"{scene}_{shard_id}.json"` 命名的索引分片文件。

公开 API：
    - load_secondary_index_shards：按配置计算需要的二级索引分片，并并行加载到 /dev/shm，返回缓存目录
"""

import hashlib
import json
import math
import pickle
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

from .remote_client import create_remote_client

__all__ = [
    "get_secondary_index_tmp_dir",
    "build_shared_secondary_index_dict",
    "load_secondary_index_entry",
    "load_scene_map_shards",
    "load_date_split_shards",
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


def _get_shm_dir(remote_tmp_dir: Union[str, List[str]]) -> str:
    """基于远端目录的 hash 生成 /dev/shm 下的唯一子目录。list 时用 | 拼接后 hash。"""
    key = "|".join(sorted(remote_tmp_dir)) if isinstance(remote_tmp_dir, list) else remote_tmp_dir
    path_hash = hashlib.md5(key.encode()).hexdigest()
    return os.path.join("/dev/shm", "secondary_index", path_hash)

# 二级索引列表按条数分块写入 shm 时每块大小（条）
SECONDARY_INDEX_CHUNK_SIZE = 2000

def load_date_split_shards(
    remote_tmp_dir: Union[str, List[str]],
    data_client,
    split_file: Optional[Union[str, List[str]]] = None,
    is_train: bool = True,
):
    """
    load the date split shards
    1. 如果 split_file 不为空: 则直接使用 split_file 的路径加载 date_split.json
    2. 如果 split_file 为空: 从共享内存中加载 -> 遍历数据根目录查找 -> 通过 data_batch.json 查找并加载 date_split.json
    """
    remote_tmp_dir = remote_tmp_dir if isinstance(remote_tmp_dir, list) else [remote_tmp_dir]
    shm_dir = _get_shm_dir(remote_tmp_dir)
    data_batch_list_path = os.path.join(shm_dir, "data_batch_list.pkl")

    merged_batch_paths = []
    if split_file: 
        if isinstance(split_file, list):
            merged_batch_paths.extend(split_file)
        else:
            merged_batch_paths.append(split_file)
    elif os.path.exists(data_batch_list_path):
        with open(data_batch_list_path, "rb") as f:
            batch_paths = pickle.load(f)
        merged_batch_paths.extend(batch_paths)
    
    if len(merged_batch_paths) == 0:
        for remote_root in remote_tmp_dir:
            normalized = remote_root.rstrip("/")
            if os.path.basename(normalized) == "scene":
                remote_root = os.path.dirname(normalized)
            date_split_path = os.path.join(remote_root, "date_split.json")
            if data_client.contains(date_split_path):
                merged_batch_paths.append(remote_root)
                continue
            remote_path = os.path.join(remote_root, "data_batch.json")
            if not data_client.contains(remote_path):
                merged_batch_paths.append(remote_root)
                continue
            content = data_client.load_json(remote_path)
            if isinstance(content, list):
                batch_paths = content
            elif isinstance(content, dict):
                batch_paths = content.get("batches", content.get("paths", []))
            else:
                batch_paths = []
            merged_batch_paths.extend(batch_paths)

    # 遍历 merged_batch_paths，下载各路径下的 date_split.json，融合为 remote_date_split
    remote_date_split = {}
    duplicate_dates = set()
    for batch_path in merged_batch_paths:
        if batch_path.endswith('date_split.json'):
            date_split_path = batch_path
        else:
            date_split_path = os.path.join(batch_path.rstrip("/"), "date_split.json")
        loaded = data_client.load_json(date_split_path)
        duplicate_dates |= loaded.keys() & remote_date_split.keys()
        remote_date_split.update(loaded)
    if duplicate_dates:
        print(f"[SecondaryIndexV2][WARNING] date_split merge {len(duplicate_dates)} duplicate dates: {sorted(duplicate_dates)[:5]} ...", flush=True)
    return remote_date_split

def load_scene_map_shards(
    remote_tmp_dir: Union[str, List[str]],
    data_client,
    data_config,
    scene_extract_override,
    is_train: bool = True,
):
    """
    load the scene map shards from /dev/shm
    """
    remote_tmp_dir = remote_tmp_dir if isinstance(remote_tmp_dir, list) else [remote_tmp_dir]
    shm_dir = _get_shm_dir(remote_tmp_dir)
    shm_scene_map = os.path.join(shm_dir, "scene_map.pkl")
    if os.path.exists(shm_scene_map):
        with open(shm_scene_map, "rb") as f:
            return pickle.load(f)
    
    print(f"[SecondaryIndexV2][WARNING] scene map not found in shm, load scene map from remote.")

    scene_map, _ = _collect_required_secondary_index_filenames(
        data_client=data_client,
        remote_tmp_dirs=remote_tmp_dir,
        scene_extract_cfg=_resolve_scene_extract_cfg(data_config, is_train, scene_extract_override),
        max_frame=data_config.get("max_frame", float("inf")),
    )
    return scene_map

def load_secondary_index_shards(
    data_config,
    is_train: bool,
    enabled: bool = True,
    scene_extract_override=None,
    force_reload: bool = False,
) -> str:
    """
    加载需要用到的二级索引分片到 /dev/shm。

    返回：
        - 返回 /dev/shm 下的缓存目录路径
    """
    local_rank = _get_local_rank()
    is_rank0 = local_rank == 0
    rank_tag = f"local_rank={local_rank}"

    start_time = time.time()
    mode = "train" if is_train else "eval"
    print(f"[SecondaryIndexV2][{rank_tag}] start: load_secondary_index_shards (mode={mode})", flush=True)

    remote_tmp_dir = get_secondary_index_tmp_dir(data_config, is_train)
    remote_tmp_dir = remote_tmp_dir if isinstance(remote_tmp_dir, list) else [remote_tmp_dir]

    data_client = create_remote_client(data_config, cluster_path=None)

    shm_dir = _get_shm_dir(remote_tmp_dir)
    os.makedirs(shm_dir, exist_ok=True)

    shm_scene_map_path = os.path.join(shm_dir, "scene_map.pkl")
    shm_scene_map_tmp_path = shm_scene_map_path + ".tmp"

    if (not force_reload) and os.path.exists(shm_scene_map_path):
        return shm_dir

    for p in (shm_scene_map_path, shm_scene_map_tmp_path):
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    scene_map = None
    if is_rank0:
        try:
            # 加载 scene_map
            scene_map, data_batch_list = _collect_required_secondary_index_filenames(
                data_client=data_client,
                remote_tmp_dirs=remote_tmp_dir,
                scene_extract_cfg=_resolve_scene_extract_cfg(data_config, is_train, scene_extract_override),
                max_frame=data_config.get("max_frame", float("inf")),
            )

            # 保存 data_batch_list
            data_batch_list_path = os.path.join(shm_dir, "data_batch_list.pkl")
            data_batch_list_tmp_path = data_batch_list_path + ".tmp"
            data_batch_list_flat = [p for batch_list in data_batch_list for p in batch_list]
            with open(data_batch_list_tmp_path, "wb") as f:
                pickle.dump(data_batch_list_flat, f)
                f.flush()
            os.rename(data_batch_list_tmp_path, data_batch_list_path)

            # 保存 scene_map
            with open(shm_scene_map_tmp_path, "wb") as f:
                pickle.dump(scene_map, f)
                f.flush()
            os.rename(shm_scene_map_tmp_path, shm_scene_map_path)
        except Exception as e:
            print(f"[SecondaryIndexV2][{rank_tag}] load scene_map error: {e}")
            raise e
    else:
        for _ in range(3000):
            if os.path.exists(shm_scene_map_path):
                try:
                    with open(shm_scene_map_path, "rb") as f:
                        scene_map = pickle.load(f)
                except OSError:
                    pass
                break
            time.sleep(0.1)

    if scene_map is None:
        raise FileNotFoundError(f"[SecondaryIndexV2][{rank_tag}] load scene_map failed")

    if enabled:
        required_files: List[Tuple[str, str]] = []
        for scene, cum_path_list in scene_map.items():
            for idx, (_cum, scene_path, *_) in enumerate(cum_path_list):
                required_files.append((scene_path, f"{scene}_{idx}.pkl"))

        _download_secondary_index_files(
            data_config=data_config,
            remote_tmp_dir=remote_tmp_dir,
            required_filenames=required_files,
            shm_dir=shm_dir,
        )

    elapsed_time = time.time() - start_time
    print(
        f"[SecondaryIndexV2][{rank_tag}] done: load_secondary_index_shards (mode={mode}), "
        f"elapsed={elapsed_time:.2f}s",
        flush=True,
    )

    return shm_dir

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
            try:
                dist.barrier()
                if dist.get_rank() == 0:
                    mode = "train" if is_train else "eval"
                    print(f"[SecondaryIndexV2] all ranks ready (mode={mode})", flush=True)
            except Exception:
                dist.barrier()
                raise
    except Exception:
        # 没有 torch / 或 dist 不可用：直接跳过同步
        pass

    return result


def load_secondary_index_entry(
    scene: str,
    idx: int,
    prefetched_shm_dir: str,
    data_client,
    scene_map: Dict[str, List],
    *,
    verbose: bool = False,
) -> Any:
    """
    加载单条二级索引分片并返回其 payload。

    读取顺序：
    1) 优先从共享内存路径加载
    2) 缺失则通过远程路径下载
    """
    if scene not in scene_map:
        raise ValueError(f"[SecondaryIndexV2][ERROR] scene {scene} not found in scene_map")

    cum_path_list = scene_map[scene]
    if len(cum_path_list) == 0 or cum_path_list[-1][0] == 0:
        raise ValueError(f"[SecondaryIndexV2][ERROR] scene {scene} num of frames is 0 but idx={idx}")

    total_frames = cum_path_list[-1][0]
    idx = idx % total_frames if idx >= total_frames else idx

    shard_index = next((i for i, entry in enumerate(cum_path_list) if idx < entry[0]), len(cum_path_list) - 1)
    real_idx = idx - (cum_path_list[shard_index - 1][0] if shard_index > 0 else 0)

    filename = f"{scene}_{shard_index}.pkl"
    shm_path = os.path.join(prefetched_shm_dir, filename)
    chunk_index = real_idx // SECONDARY_INDEX_CHUNK_SIZE
    sub_idx = real_idx % SECONDARY_INDEX_CHUNK_SIZE
    chunk_path = f"{shm_path}.{chunk_index}"
    if os.path.exists(chunk_path):
        with open(chunk_path, "rb") as f:
            payload = pickle.load(f)
            return payload[sub_idx]
    if os.path.exists(shm_path):
        with open(shm_path, "rb") as f:
            payload = pickle.load(f)
            return payload[real_idx]

    if verbose:
        print(f"[SecondaryIndexV2][WARNING] 索引文件 {filename} 不在共享内存中，回退到文件系统加载", flush=True)

    scene_path = cum_path_list[shard_index][1]
    payload = data_client.load_json(scene_path)
    if payload is None or len(payload) == 0:
        raise ValueError(f"[SecondaryIndexV2][ERROR] scene file {scene_path} is empty")
    return payload[real_idx]


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
    remote_tmp_dirs: List[str],
    scene_extract_cfg,
    max_frame,
):
    """
    计算需要加载的二级索引文件名。

    - remote_tmp_dirs 支持多个目录，内部统一按 list 处理。
    - 如果存在 data_batch.json，则按批次处理每个 batch 路径，汇总 scene_num.json，并结合 scene_extract_cfg、max_frame 做过滤与采样。
    - 如果不存在 data_batch.json，则回退为单目录下的 scene_num.json 处理逻辑。

    返回: (scene_map, data_batch_list)
    """
    try:
        max_frame = float(max_frame)
    except (TypeError, ValueError):
        max_frame = float("inf")

    # Step1: 遍历 remote_tmp_dir 下载 data_batch.json
    data_batch_list: list = []
    for remote_root in remote_tmp_dirs:
        normalized = remote_root.rstrip("/")
        if os.path.basename(normalized) == "scene":
            remote_root = os.path.dirname(normalized)
        remote_path = os.path.join(remote_root, "data_batch.json")
        if not data_client.contains(remote_path):
            data_batch_list.append([remote_root])
            continue
        content = data_client.load_json(remote_path)
        if isinstance(content, list):
            batch_paths = content
        elif isinstance(content, dict):
            batch_paths = content.get("batches", content.get("paths", []))
        else:
            batch_paths = []
        data_batch_list.append(batch_paths)

    # Step2: 各 batch 路径下载 scene_num.json
    batch_scene_nums: list = []
    for idx, batch_list in enumerate(data_batch_list):
        for batch_path in batch_list:
            scene_num_path = None
            for candidate_dir in (
                batch_path,
                os.path.join(batch_path, "scene_statistics")
            ):
                candidate = os.path.join(candidate_dir, 'scene_num.json')
                if data_client.contains(candidate):
                    scene_num_path = candidate
                    break
            if scene_num_path is None:
                raise FileNotFoundError(f"scene_num.json not found under {batch_path} directory")
            scene_num_dict = data_client.load_json(scene_num_path)
            batch_scene_nums.append((batch_path, scene_num_dict, idx))

    # Step3: 计算每个场景在每个 batch 中的累计物理样本数和 scene_path
    tmp_scene_map: dict = {}
    for batch_path, scene_num, idx in batch_scene_nums:
        for scene, extract_rate in scene_extract_cfg.items():
            if scene not in scene_num:
                continue
            if scene_num[scene] == 0:
                continue
            if scene not in tmp_scene_map:
                tmp_scene_map[scene] = []
            prev_cum = tmp_scene_map[scene][-1][0] if len(tmp_scene_map[scene]) > 0 else 0
            tmp_scene_map[scene].append((prev_cum + scene_num[scene], os.path.join(batch_path, "scene", f"{scene}.json"), idx))

    # Step4: 汇总到 scene_map ： scene -> [(cum_physical, scene_path), ...]
    consumed = 0
    scene_map: dict = {}
    for scene, extract_rate in scene_extract_cfg.items():
        if scene not in tmp_scene_map:
            continue

        total_scene_num = tmp_scene_map[scene][-1][0]
        total_frames = int(total_scene_num / extract_rate)
        if total_frames == 0:
            continue

        remaining = float("inf") if math.isinf(max_frame) else max_frame - consumed
        if remaining <= 0:
            break

        use_frames = min(total_frames, remaining)
        if scene not in scene_map:
            scene_map[scene] = []
        physical_use_frames = math.ceil(use_frames * extract_rate)
        for idx, item in enumerate(tmp_scene_map[scene]):
            if idx > 0 and tmp_scene_map[scene][idx - 1][0] > physical_use_frames:
                break
            scene_map[scene].append(item)
        
        consumed += use_frames

    return scene_map, data_batch_list


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

        # 如果是 shm 模式且文件已存在（单文件或分块格式的首块），跳过
        if shm_dir:
            shm_path = os.path.join(shm_dir, filename)
            if os.path.exists(shm_path) or os.path.exists(f"{shm_path}.0"):
                return filename, None

        try:
            payload = _MP_REMOTE_CLIENT.load_json(remote_path)

            # 存储到 shm：按 SECONDARY_INDEX_CHUNK_SIZE 分块，写为 filename.0, filename.1, ...
            if shm_dir:
                shm_path = os.path.join(shm_dir, filename)
                for i in range(0, len(payload), SECONDARY_INDEX_CHUNK_SIZE):
                    chunk = payload[i : i + SECONDARY_INDEX_CHUNK_SIZE]
                    chunk_path = f"{shm_path}.{i // SECONDARY_INDEX_CHUNK_SIZE}"
                    with open(chunk_path, "wb") as f:
                        pickle.dump(chunk, f)

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
    required_filenames: List[Tuple[str, str]],
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
            f"[SecondaryIndexV2] SHM Mode: total {len(required_filenames)} files, "
            f"rank {local_rank} handles {len(my_required_filenames)}",
            flush=True,
        )

    download_tasks = list(my_required_filenames)

    if not download_tasks:
        if is_rank0 and not required_filenames:
            print(f"[SecondaryIndexV2][WARNING] rank {local_rank} has no secondary index files to load")
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
                desc=f"SecondaryIndexV2(R{local_rank})",
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
                        f"[SecondaryIndexV2][{local_rank}] progress: {completed}/{len(download_tasks)} files, {rate:.1f} files/s",
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
            f"[SecondaryIndexV2][{local_rank}] completed with errors: "
            f"{completed}/{len(download_tasks)} files, elapsed={elapsed:.2f}s, "
            f"avg={rate:.1f} files/s, errors={error_count}",
            flush=True,
        )
        print(
            f"[SecondaryIndexV2][{local_rank}] sample_errors (up to 5): {sample_errors}",
            flush=True,
        )


