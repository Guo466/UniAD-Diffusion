import os
import sys
import pickle
import json
import hashlib
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from .remote_client import RemoteClient, create_remote_client

__all__ = [
    "build_frame_label_dict",
    "load_frame_label",
]

# 子进程内复用 RemoteClient，避免每个文件都重新创建 client/连接池
_MP_REMOTE_CLIENT = None
_MP_REMOTE_CLIENT_PID = None

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
    """基于远端目录的 hash 生成 /dev/shm 下的唯一子目录。"""
    key = "|".join(sorted(remote_tmp_dir)) if isinstance(remote_tmp_dir, list) else remote_tmp_dir
    path_hash = hashlib.md5(key.encode()).hexdigest()
    return os.path.join("/dev/shm", "frame_label_map", path_hash)

def _get_date_split_cache_path(shm_dir: str, is_train: bool) -> str:
    return os.path.join(shm_dir, "data_mining_train.json" if is_train else "data_mining_eval.json")

def _get_rank0_done_path(shm_dir: str, is_train: bool) -> str:
    """Rank0 完成标志文件路径，供多进程共享：rank0 写完后其它进程可检测到。"""
    return os.path.join(shm_dir, "data_mining_train.rank0_done" if is_train else "data_mining_eval.rank0_done")

def _load_date_mining_from_data_dirs(
    raw_data_dir: Union[str, List[str]],
    data_client: RemoteClient,
) -> Dict[str, str]:
    """从 raw_data_dir（单个或列表）下的各目录加载 date_mining.json 并合并。"""
    result: Dict[str, str] = {}
    dirs = [raw_data_dir] if isinstance(raw_data_dir, str) else raw_data_dir
    for rd in dirs:
        normalized = rd.rstrip("/")
        if os.path.basename(normalized) == "scene":
            normalized = os.path.dirname(normalized)
        path = os.path.join(normalized, "date_mining.json")
        result.update(data_client.load_json(path))
    return result

def _dump_data(path: str, payload: Any) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def _load_frame_label_partition_inflight_mp(args):
    """
    子进程内加载一整份分配到该进程的任务列表（partition），并在进程内用线程做 in-flight 并发。
    结果存储到 shm_dir。
    """
    tasks, data_config, threads_per_proc, shared_counter, shared_lock, shm_dir, force_reload = args

    # 复用同一进程内的 RemoteClient（连接池）
    global _MP_REMOTE_CLIENT, _MP_REMOTE_CLIENT_PID
    pid = os.getpid()
    if _MP_REMOTE_CLIENT is None or _MP_REMOTE_CLIENT_PID != pid:
        _MP_REMOTE_CLIENT = create_remote_client(data_config)
        _MP_REMOTE_CLIENT_PID = pid

    if shm_dir:
        os.makedirs(shm_dir, exist_ok=True)

    errors: List[str] = []

    def _load_one(task):
        date, remote_path = task
        shm_path = os.path.join(shm_dir, date)

        # 如果文件已存在且不强制重载，跳过
        if (not force_reload) and os.path.exists(shm_path):
            return date, None

        try:
            if not _MP_REMOTE_CLIENT.contains(remote_path):
                return date, None

            payload: Dict[str, Any] = _MP_REMOTE_CLIENT.load_json(remote_path)
            _dump_data(shm_path, payload)
            return date, None
        except Exception as e:
            return date, e

    # 计数器更新：减少锁竞争，按批次累加
    local_completed = 0

    def _bump_counter(n: int):
        if shared_counter is None or shared_lock is None or n <= 0:
            return
        with shared_lock:
            shared_counter.value = int(shared_counter.value) + int(n)

    workers = threads_per_proc if (threads_per_proc > 1 and len(tasks) > 1) else 1
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as tp:
            futures = [tp.submit(_load_one, t) for t in tasks]
            for fut in as_completed(futures):
                date, err = fut.result()
                if err is not None:
                    errors.append(f"{date}: {err}")
                local_completed += 1
                if local_completed % 50 == 0:
                    _bump_counter(50)
    else:
        for t in tasks:
            date, err = _load_one(t)
            if err is not None:
                errors.append(f"{date}: {err}")
            local_completed += 1
            if local_completed % 50 == 0:
                _bump_counter(50)

    # flush remainder
    rem = local_completed % 50
    if rem:
        _bump_counter(rem)

    return local_completed, errors


def load_frame_label_dict(
    shm_dir: str,
    pending_dates: List[Tuple[str, str]],
    data_config: Mapping[str, Any],
    num_workers: int = 8,
    force_reload: bool = False,
) -> str:
    local_rank = _get_local_rank()
    local_world_size = _get_local_world_size()
    is_rank0 = local_rank == 0
    rank_tag = f"local_rank={local_rank}"

    start_time = time.time()
    print(f"[FrameLabel][{rank_tag}] start: load_frame_label_shards", flush=True)

    my_pending_dates = [
        d for i, d in enumerate(pending_dates) if i % local_world_size == local_rank
    ]
    if is_rank0:
        print(
            f"[FrameLabel] SHM Mode: total {len(pending_dates)} dates, "
            f"rank {local_rank} handles {len(my_pending_dates)}",
            flush=True,
        )

    download_tasks = [(date, remote_path) for date, remote_path in my_pending_dates]

    if not download_tasks:
        elapsed_time = time.time() - start_time
        print(
            f"[FrameLabel][{rank_tag}] done: load_frame_label_shards (no tasks), elapsed={elapsed_time:.2f}s",
            flush=True,
        )
        return shm_dir

    # 多进程配置
    mp_workers = int(os.environ.get("FRAME_LABEL_MULTIPROC_WORKERS", "8"))
    mp_workers = min(mp_workers, len(download_tasks))

    threads_per_proc = int(os.environ.get("FRAME_LABEL_THREADS_PER_PROC", "8"))
    threads_per_proc = max(1, threads_per_proc)

    # 将任务分配到各个进程
    partitions: List[List[tuple]] = [[] for _ in range(mp_workers)]
    for i, task in enumerate(download_tasks):
        partitions[i % mp_workers].append(task)

    completed = 0
    last_print_time = start_time

    pbar = None
    if is_rank0:
        try:
            lightning_like_bar_format = (
                "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            )
            pbar = tqdm(
                total=len(download_tasks),
                desc=f"FrameLabel(R{local_rank})",
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
                _load_frame_label_partition_inflight_mp,
                (part, data_config, threads_per_proc, shared_counter, shared_lock, shm_dir, force_reload),
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
                if completed % 100 == 0 or completed >= len(download_tasks) or (now - last_print_time) >= 5.0:
                    elapsed = now - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(
                        f"[FrameLabel] rank {local_rank} progress: {completed}/{len(download_tasks)} dates, {rate:.1f} dates/s",
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

    elapsed_time = time.time() - start_time
    rate = len(my_pending_dates) / elapsed_time if elapsed_time > 0 else 0

    if error_count > 0:
        print(
            f"[FrameLabel][{rank_tag}][ERROR] completed with errors: "
            f"{len(my_pending_dates)}/{len(download_tasks)} dates, elapsed={elapsed_time:.2f}s, "
            f"avg={rate:.1f} dates/s, errors={error_count}",
            flush=True,
        )
        print(
            f"[FrameLabel][{rank_tag}] sample_errors (up to 5): {sample_errors}",
            flush=True,
        )
    else:
        print(
            f"[FrameLabel][{rank_tag}] done: load_frame_label_shards, elapsed={elapsed_time:.2f}s, "
            f"avg={rate:.1f} dates/s",
            flush=True,
        )

    return shm_dir

def build_frame_label_dict(
    data_config: Mapping[str, Any],
    is_train: bool = True,
    force_reload: bool = False,
) -> str:
    data_client = create_remote_client(data_config)
    local_rank = _get_local_rank()
    is_rank0 = local_rank == 0

    # 仅下载一次 date_split，多进程共享缓存
    raw_data_dir = data_config["tmp_data_dir"] if is_train else data_config["tmp_data_dir_test"]
    shm_dir = _get_shm_dir(raw_data_dir)
    os.makedirs(shm_dir, exist_ok=True)
    date_split_cache = _get_date_split_cache_path(shm_dir, is_train)
    if (not force_reload) and os.path.exists(date_split_cache):
        with open(date_split_cache, "rb") as f:
            date_split = pickle.loads(f.read())
    else:
        rank0_done_path = _get_rank0_done_path(shm_dir, is_train)
        # 在 rank0 执行前清除旧标志，避免其它 rank 误判（所有进程进入此分支时统一清一次）
        if os.path.exists(rank0_done_path):
            try:
                os.remove(rank0_done_path)
            except OSError:
                pass
        if is_rank0:
            date_split = {}
            try:
                date_split = _load_date_mining_from_data_dirs(raw_data_dir, data_client)
                _dump_data(date_split_cache, date_split)
                print(f"[FrameLabel] rank0: downloaded {'train' if is_train else 'eval'} data_mining to {date_split_cache}", flush=True)
            except Exception as e:
                print(f"[FrameLabel] rank0: downloaded {'train' if is_train else 'eval'} data_mining error : {e}")
            try:
                with open(rank0_done_path, "w"):
                    pass
            except OSError:
                pass
        else:
            # 等待 rank0 写入缓存或 rank0_done 标志，最多等待 120s，兜底再自行下载
            date_split = None
            for _ in range(1200):
                if os.path.exists(rank0_done_path):
                    try:
                        with open(date_split_cache, "rb") as f:
                            date_split = pickle.loads(f.read())
                    except Exception:
                        pass
                    break
                time.sleep(0.1)
    
    if not data_config.get("load_frame_label_shards", True):
        return ""
    
    if not date_split:
        print(f"[FrameLabel][ERROR] rank{local_rank}: date_mining is None!")
        return ""

    use_tos_data = data_config.get("use_tos_data", False)
    pending_dates = []
    for date, value in date_split.items():
        remote_dir = os.path.join("tos://" if use_tos_data else "s3://", value)
        pending_dates.append((date, os.path.join(remote_dir, date, "frame_scene.json"))) 

    # 加载 frame_label 到共享内存（使用多进程模式）
    result = load_frame_label_dict(
        shm_dir=shm_dir,
        pending_dates=pending_dates,
        data_config=data_config,
        num_workers=int(data_config.get("frame_label_workers", 8)),
        force_reload=force_reload,
    )

    # 等待所有 rank 完成预取，避免训练阶段不同步
    try:
        import torch.distributed as dist  # type: ignore
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            if dist.get_rank() == 0:
                mode = "train" if is_train else "eval"
                print(f"[FrameLabel] all ranks loaded frame_label shards (mode={mode})", flush=True)
    except Exception:
        pass

    return result

def load_frame_label(
    shm_dir: str,
    date: str,
    frame_id: str,
    is_train: bool = True,
    data_client: Optional[RemoteClient] = None,
    use_tos_data: bool = False,
) -> Optional[Any]:
    """
    读取指定 date + frame_id 的单帧标签, 优先从 shm 目录读取, 缺失则从远端加载
    """
    if shm_dir:
        fpath = os.path.join(shm_dir, date)
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                payload = pickle.loads(f.read())
            if frame_id in payload:
                return payload[frame_id]
            elif payload is not None and len(payload) > 0:
                # print(f"[FrameLabel][ERROR] frame {date} {frame_id} not found in payload")
                return None

    return None

def load_date_mining(
    shm_dir: str,
    raw_data_dir: Union[str, List[str]],
    data_client: RemoteClient,
    is_train: bool = True,
) -> Optional[Dict[str, str]]:
    if shm_dir:
        # 有缓存时不加载 date_mining
        return None
    if not shm_dir:
        shm_dir = _get_shm_dir(raw_data_dir)
    date_split_cache = _get_date_split_cache_path(shm_dir, is_train)
    if os.path.exists(date_split_cache):
        with open(date_split_cache, "rb") as f:
            return pickle.loads(f.read())
    
    print(f"[FrameLabel][WARNING] date_mining not found in shm, load date_mining from remote.")

    return _load_date_mining_from_data_dirs(raw_data_dir, data_client)
