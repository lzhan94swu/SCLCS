import os
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import Dataset


class TSWdataset(Dataset): # 时间窗口数据集，用于MDD分类和Sub识别，根据任务输出类别标签和样本ID（含窗口编号）
    def __init__(self, task, id_col='SubID', class_col='if_mdd', window_size=70, stride=35):
        sub_info_path = './CoreSet904_458MDDvs446NC.csv'
        ts_mat_dir = './data/ROISignals_FunImgARglobalCWF/'
        dmn_in_dosen = [1, 4, 5, 6, 7, 11, 13, 14, 15, 17, 20, 25, 63, 72, 73, 85, 90, 91, 93, 94, 100, 102, 104, 105, 108, 111, 112, 115, 117, 124, 132, 134, 137]
        roi_idx = [i-1+1408 for i in dmn_in_dosen] # Dosenbach atlas is 1409-1568 columns, idx starts from 0
        self.task = task
        self.samples = []
        sub_info = pd.read_csv(sub_info_path)
        self.sub_list = sub_info[id_col].unique().tolist()
        self.sub2idx = {sub: idx for idx, sub in enumerate(self.sub_list)}
        for _, sub_row in sub_info.iterrows():
            sub_id = sub_row[id_col] # subject ID, str
            class_label = sub_row[class_col]
            class_label = 0 if class_label == -1 else 1 # -1 -> 0, 1 -> 1
            sub_idx = self.sub2idx[sub_id]
            ts_mat_path = os.path.join(ts_mat_dir, f'ROISignals_{sub_id}.mat')
            ts = sio.loadmat(ts_mat_path)['ROISignals']
            ts = ts[:, roi_idx] # rows: time points, columns: ROIs
            T, D = ts.shape
            # windowing
            for start in range(0, T - window_size + 1, stride):
                sw_id = int(start/stride)+1
                sub_swID = f'{sub_id}_SW{sw_id}'
                window = ts[start:start + window_size]
                window = window.T # shape: (D, window_size)
                assert window.shape == (len(roi_idx), window_size), f"Window shape mismatch: {window.shape} != ({window_size}, {len(roi_idx)})"
                window = np.expand_dims(window, axis=0) # shape: (1, D, window_size)
                self.samples.append((window, sub_idx, class_label, sub_swID))
            self.wid2index = {swid: idx for idx, (_, _, _, swid) in enumerate(self.samples)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window, sub_idx, class_lable, sub_wid = self.samples[idx]
        x = torch.tensor(window, dtype=torch.float32)
        if self.task == 'mddC':
            y = torch.tensor(class_lable, dtype=torch.long)
        elif self.task == 'subC':
            y = torch.tensor(sub_idx, dtype=torch.long)
        else:
            raise ValueError(f"Unknown task: {self.task}")
        return x, y, sub_wid