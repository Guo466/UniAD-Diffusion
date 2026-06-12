import os
import numpy as np
import torch
import json
import pickle
from datetime import datetime

def img2real(img_pts, resolution, i0, j0):
    """
    img_pts:
        - np.ndarray / torch.Tensor / list, shape (..., 2)
        - or List[above], each shape (Ni, 2)

    return:
        - same structure as input
        - real-world coords (x, y), in meters
        - NOT torch.Tensor
    """

    # ---- recursive case: list of polygons ----
    if isinstance(img_pts, (list, tuple)):
        return [
            img2real(pts, resolution, i0, j0)
            for pts in img_pts
        ]

    # ---- to numpy ----
    if isinstance(img_pts, torch.Tensor):
        pts = img_pts.detach().cpu().numpy()
    else:
        pts = np.asarray(img_pts, dtype=np.float32)

    i, j = pts[..., 1], pts[..., 0]
    x = (j - j0) * resolution
    y = (i0 - i) * resolution   # image i 向下，real y 向上

    return np.stack([x, y], axis=-1) 

def ndarray2json(ndarray, save_path, mode):
    ndarray_list = ndarray.tolist()
    save_path = os.path.join(save_path, f"{mode}.json")
    with open(save_path, 'w') as f:
        json.dump(ndarray_list, f)

def save_bad_batch(model_input, loss_for_check, rank, cur_epoch, cur_step, cur_batch_idx, save_root):
        """Save bad batch data when loss exceeds threshold.
        
        Args:
            model_input: Input dict to the model
            loss_for_check: Detached scalar loss for threshold checking
            rank: Current process rank
            cur_epoch: Current training epoch
            t: Timestep tensor
        """
        
        save_dir = os.path.join(save_root, "saved_bad_batches")
        os.makedirs(save_dir, exist_ok=True)
        
        
        
        # create filename with rank, loss, and timestamp for easy identification
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        save_name = (
            f"bad_batch_epoch{cur_epoch}_step{cur_step}_batch{cur_batch_idx}"
            f"_rank{rank}_loss{loss_for_check:.4f}_{timestamp}.pkl"
        )
        save_path = os.path.join(save_dir, save_name)
        
        def _to_cpu(x):
            """Convert tensor to CPU, leave other types unchanged."""
            if isinstance(x, torch.Tensor):
                return x.detach().cpu()
            return x
        
        # convert model_input to CPU for saving
        cpu_model_input = (
            {k: _to_cpu(v) for k, v in model_input.items()}
            if isinstance(model_input, dict)
            else model_input
        )
        
        payload = {
            "loss_for_check": float(loss_for_check.item()),
            "model_input": cpu_model_input,
        }
        with open(save_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    