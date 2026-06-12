import torch
import os
import argparse


def convert_rl_checkpoint_strict(ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # support both direct state_dict and PL checkpoints
    state_dict = ckpt.get("state_dict", ckpt)
    # state_dict = {k.replace('model.', ''):v for k,v in state_dict.items()}
    
    for k in state_dict.keys():
        if k.startswith("model.policy"):
            print(k)
    policy_state_dict = {
        k.replace("policy_model.", ""): v
        for k, v in state_dict.items()
        if k.startswith("model.policy_model")
    }
    print("policy_state_dict.keys()", policy_state_dict.keys())

    new_ckpt_path = os.path.splitext(ckpt_path)[0] + "_converted.ckpt"
    print(f"Saving converted checkpoint to {new_ckpt_path}")
    ckpt["state_dict"] = policy_state_dict
    torch.save(ckpt, new_ckpt_path)
    

model_path = "./lightning_logs/version_20/checkpoints/m24_rl.ckpt"
convert_rl_checkpoint_strict(model_path)