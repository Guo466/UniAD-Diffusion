# -*- coding: utf-8 -*-
import pickle
import os

data_path = '/mnt/afs/yuqiushuang/workspace/dif_wip1/pl_diffusion_models/4011520_mean_std_seperate_5s_5hz.pkl'
data_txt_path = '/mnt/afs/yuqiushuang/workspace/dif_wip1/pl_diffusion_models/4011520_mean_std_seperate_5s_5hz.txt'

with open(data_path, 'rb') as f:
    data = pickle.load(f)
# read pkl data    
import pdb;pdb.set_trace()
# key_list 顺序很重要，一般不修改，只在后面增量添加，修改后要和sdk同步
key_list = ['ego_future_status_x', 'ego_future_status_y', 'ego_future_status_fixed_x', 'ego_future_status_fixed_y', 'ego_delta_yaw', 'ego_fixed_delta_yaw']
with open(data_txt_path, 'w', encoding='utf-8') as txt_file:
    for key in key_list:
        mean = " ".join(map(str,data[key]['mean'].tolist()))
        std = " ".join(map(str,data[key]['std'].tolist()))
        txt_file.write(f'{mean}\n')
        txt_file.write(f'{std}\n')
txt_file.close()

# check save txt file 
with open(data_txt_path, 'r', encoding='utf-8') as read_file:
    lines = read_file.readlines()
import pdb;pdb.set_trace()
        
# 可视化
import pickle
import matplotlib.pyplot as plt
# 读取 pkl 文件
with open(data_path, 'rb') as f:
    data = pickle.load(f)
print(data)

vis_folder = f'{data_path.split("/")[-1].split(".")[0]}_vis'
if not os.path.exists(vis_folder):
    os.makedirs(vis_folder)
for k, v in data.items():
    print(k, v['mean'].shape, v['std'].shape)
    plt.plot(v['mean'])
    plt.savefig(f'{vis_folder}/{k}_mean.png')
    plt.clf()
    plt.plot(v['std'])
    plt.savefig(f'{vis_folder}/{k}_std.png')
    plt.clf()