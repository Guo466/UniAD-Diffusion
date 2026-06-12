import os
import sys
import time
import struct
import numpy as np

import torch
import torch.nn.functional as F

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

def logger(level, msg):
    timestamp = time.strftime("%H:%M:%S", time.localtime(time.time()))
    if level == "INFO" or level == "I":
        log = "\033[1;32m{} [{}] {}\033[0m".format(timestamp, level, msg)
    elif level == "WARNNING" or level == "W":
        log = "\033[1;33m{} [{}] {}\033[0m".format(timestamp, level, msg)
    elif level == "FATAL" or level == "ERROR" or level == "E":
        log = "\033[1;31m{} [{}] {}\033[0m".format(timestamp, level, msg)
    else:
        log = "\033[1;32m{} [{}] {}\033[0m".format(timestamp, level, msg)
    print(log)

## TODO You need config these for different dlp version
INPUT_SHAPE = {
    "agent_status": [1, 50, 21, 6],
    "agent_attrs": [1, 50, 3],
    "agent_time_mask": [1, 50, 21],
    "laneline_pts": [1, 50, 200, 2],
    "laneline_attrs": [1, 50, 3],
    "laneline_mask": [1, 50],
    "ego_curr_status": [1, 4],
    "category_feature": [50 + 1, 256],
    "traffic_light_feature": [50 + 1, 256],
    "car_light_feature": [50 + 1, 256],
    "polygon_color_feature": [50, 256],
    "polygon_laneline_type_feature": [50, 256],
    "polygon_laneline_style_feature": [50, 256],
    # "polygon_other_type_feature": [50, 128],
    "navitopo_pts": [1, 5, 200, 2],
    "navitopo_mask": [1, 5],
    # "agent_query": [1, 1, 51, 128],
    "sample_steps": [1],
    "timestamp_ns": [1],
    "input_noise": [8, 1, 25, 3],
    "input_noise_fix_distance": [8, 1, 40, 3],
    "mean_std": [1, 6, 25],
    "mean_std_fixed": [1, 6, 40],
    "occ_polygons_pts": [1, 80, 100, 2],
    "occ_polygons_mask": [1, 80],
    "occ_polygons_attrs": [1, 80, 5],
}

OUTPUT_SHAPE = {
    "trajectory": [8, 25, 5],
    # "probability": [50],
    "trajectory_fixed": [8, 40, 5],
    # "probability_fixed": [50],
    # "agent_prediction_multimode_cls": [50,6],
    # "agent_prediction_multimode_reg": [50,6,80,5],
    # "agent_prediction": [50, 80, 5],
    # "agents_importance": [50, 1],
    # "laneline_importance": [50, 1],
}


class DumpData:
    def __init__(self, bin_dir):
        self.bin_dir = bin_dir
        self.input_datas = {}
        self.output_datas = {}
        self._read_binary_files()
        self.max_input_frame = self._calc_max_input_frame()

    def _calc_max_input_frame(self):
        max_input_frame = float("inf")
        for key, val in self.input_datas.items():
            if len(val) < max_input_frame:
                max_input_frame = len(val)
        return max_input_frame

    def _read_binary_file(self, file_name):
        with open(file_name, "rb") as file:
            logger("I", f"Read file {file_name}")
            if 'agent_prediction_multimode_reg' in file_name:
                num_shape_elements = 6
            elif 'sample_steps' in file_name or 'timestamp_ns' in file_name or 'occ_polygons_mask' in file_name:
                num_shape_elements = 3
            elif 'mean_std' in file_name or 'occ_polygons_attrs' in file_name:
                num_shape_elements = 4
            else:
                num_shape_elements = 5
            shape_data = file.read(num_shape_elements * struct.calcsize("Q"))
            shape = list(struct.unpack(f"{num_shape_elements}Q", shape_data))
            size_of_t = shape.pop()

            logger("I", f"Shape:{shape}")
            logger("I", f"Size of T:{size_of_t}")

            if size_of_t == 4:  # float
                dtype = np.float32
            elif size_of_t == 8:  # double
                dtype = np.float64
                if 'timestamp_ns' in file_name:
                    dtype = np.uint64
            else:
                raise ValueError("Unsupported data type size.")

            datas = []
            num_elements = np.prod(shape)
            while True:
                data_bytes = file.read(num_elements * size_of_t)
                if not data_bytes:
                    logger("W", "Read data end.")
                    break
                if len(data_bytes) != num_elements * size_of_t:
                    logger(
                        "W",
                        "Data read size does not match expected number of elements.",
                    )
                    break
                data = np.zeros(num_elements, dtype=dtype)
                data = np.frombuffer(data_bytes, dtype=dtype)
                datas.append(data)
            logger("I", f"Datas length: {len(datas)}")
            # if 'timestamp_ns' in file_name:
            #     np.set_printoptions(legacy=False)
            #     print(datas)
            #     assert False
            return shape, datas

    def _read_binary_files(self):
        for file in os.listdir(self.bin_dir):
            if not file.endswith(".bin"):
                continue
            filename = file.split(".")[0]
            if filename in INPUT_SHAPE.keys():
                shape, datas = self._read_binary_file(
                    os.path.join(self.bin_dir, filename + ".bin")
                )
                # self.input_datas[filename] = [
                #     torch.tensor(data, dtype=torch.float32).reshape(shape)
                #     for data in datas
                # ]
                if 'timestamp_ns' in filename:
                    self.input_datas['timestamp'] = [torch.tensor(data, dtype=torch.long) for data in datas]
                else:
                    self.input_datas[filename] = [
                        torch.tensor(data, dtype=torch.float32).reshape(
                            INPUT_SHAPE[filename]
                        )
                        for data in datas
                    ]
            if filename in OUTPUT_SHAPE.keys():
                shape, datas = self._read_binary_file(
                    os.path.join(self.bin_dir, filename + ".bin")
                )
                # self.output_datas[filename] = [
                #     torch.tensor(data, dtype=torch.float32).reshape(shape)
                #     for data in datas
                # ]
                self.output_datas[filename] = [
                    torch.tensor(data, dtype=torch.float32).reshape(
                        OUTPUT_SHAPE[filename]
                    )
                    for data in datas
                ]

    def get_input_datas(self):
        return self.input_datas

    def get_output_datas(self):
        return self.output_datas

    def iter_input_datas(self, index):
        datas = {"model_input": {}}
        if index >= self.max_input_frame:
            logger(
                "E", f"Index {index} overceed max input frame {self.max_input_frame}"
            )
            return datas
        for key in INPUT_SHAPE:
            if key == 'timestamp_ns':
                datas["model_input"]['timestamp'] = self.input_datas['timestamp'][index]
                continue
            if (
                key == "laneline_attrs" and self.input_datas[key][index].shape[2] == 4
            ):  ## TODO(qjb) adapt to roadmarker
                original_tensor = self.input_datas[key][index]
                datas["model_input"][key] = F.pad(original_tensor, pad=(0, 1), value=0)
            else:
                if key not in self.input_datas:
                    continue
                datas["model_input"][key] = self.input_datas[key][index]
        if 'ego_curr_status' in INPUT_SHAPE.keys():
            datas["model_input"]['egolight_ori'] = torch.tensor([self.input_datas['ego_curr_status'][index][0,3]],dtype=torch.float32)
        # print 
        if 'navitopo_pts' in INPUT_SHAPE.keys():
            datas["model_input"]['navitopo_pts_ori'] = torch.tensor(self.input_datas['navitopo_pts'][index][0,:,:,:],dtype=torch.float32).clone().detach()
        if 'navitopo_mask' in INPUT_SHAPE.keys():
            new_values = [0,0,0,0,0]
            datas["model_input"]['del_accLight_mask'] = torch.tensor(new_values, dtype=torch.float32)
        datas["model_input"]['map_type'] = torch.tensor([0], dtype=torch.float32)

        # 获取agent_time_mask和agent_status
        agent_time_mask = datas["model_input"]["agent_time_mask"]  # shape: [1, 50, 21]
        agent_status = datas["model_input"]["agent_status"]  # shape: [1, 50, 21, 6]
        agent_attrs = datas["model_input"]["agent_attrs"]

        # agent_time_mask[:,1:51,:] = 0.0
        # agent_status[:,1:51,:,:] = 0.0
        # agent_attrs[:,1:51,:] = 0.0
        # new_values = [-2.8005e+00, -4.7094e+00,  4.0000e-04, -2.0000e-04,  1.3100e-02,
        #      9.4580e-01]       
        # new_values = torch.tensor(new_values, dtype=torch.float32)
        # # 精度舍入
        # new_values = torch.round(new_values * (10 ** 5)) / (10 ** 5)

    
        for k,v in datas["model_input"].items():
            # if torch.isnan(v).any():
            #     print(f" {k} 中存在 NaN 值!")
            #     print(f"Tensor形状: {v.shape}")
            # print(f"{k}, {v}")
            pass
        return datas

    def iter_output_datas(self, index):
        datas = {"model_input": {}}
        if index >= self.max_input_frame:
            logger(
                "E", f"Index {index} overceed max input frame {self.max_input_frame}"
            )
            return datas
        for key in OUTPUT_SHAPE:
            data_array = self.output_datas.get(key, [])
            if index >= len(data_array):
                continue
            datas["model_input"][key] = self.output_datas[key][index]
        for k,v in datas["model_input"].items():
            # if k != "agent_prediction_multimode_cls" and k != "agent_prediction_multimode_reg":
            #     if torch.isnan(v).any():
            #         print(f" {k} 中存在 NaN 值!")
            #         print(f"Tensor形状: {v.shape}")
            # print(f"{k}, {v}")
            pass
        return datas

    def get_max_frame(self):
        return self.max_input_frame
