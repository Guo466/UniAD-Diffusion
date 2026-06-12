import gzip
import hashlib
import os
import pickle


def init_shm_cache_dir(dir_name: str = "dlp_processed_cache"):
    """初始化 /dev/shm 缓存目录，失败时返回 None。"""
    cache_dir = os.path.join("/dev/shm", dir_name)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    except Exception as e:
        print(f"[cache_utils] init shm cache dir failed: {e}")
        return None


def load_processed_cache(cache_path: str):
    """读取缓存文件，异常或不存在返回 None。"""
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        with gzip.open(cache_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[cache_utils] load shm cache failed, fallback to remote: {e}")
        return None


def save_processed_cache(cache_path: str, data, data_fea_not_emb, route_cost):
    """写入缓存文件，异常时打印提示。"""
    if not cache_path:
        return
    try:
        tmp_cache_path = f"{cache_path}.tmp"
        with gzip.open(tmp_cache_path, "wb", compresslevel=3) as f:
            pickle.dump((data, data_fea_not_emb, route_cost), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_cache_path, cache_path)
    except Exception as e:
        print(f"[cache_utils] write shm cache failed: {e}")


def get_scene_cache_file(shm_dir: str, scene: str, data_label_path: str, target_scenes):
    """构造场景缓存文件路径，非目标场景或目录为空返回 None。"""
    if not shm_dir or scene not in target_scenes:
        return None
    label_hash = hashlib.md5(data_label_path.encode("utf-8")).hexdigest()
    #     # 后缀 bump：处理缓存语义变更（如 navi_link 仅保留前方 max_dist 内），避免读到旧 pickle
    # return os.path.join(shm_dir, f"{scene}_label_{label_hash}_nla4.pkl")
    # 后缀 bump：处理缓存语义变更（如 navi_link_ded_* 独立槽位），避免读到旧 pickle
    return os.path.join(shm_dir, f"{scene}_label_{label_hash}_nla5.pkl")

