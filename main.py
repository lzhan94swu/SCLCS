import os, sys
import random
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import torch.nn as nn
import torch.optim as optim

from model import StructureModel
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

os.environ

sys.path.append("SCLCS")
from utils import TSWdataset

# 主函数中 dataloader 改为返回 index
class IndexDataLoader(DataLoader):
    def __iter__(self):
        for batch_indices in self.batch_sampler:
            batch = [self.dataset[i] for i in batch_indices]
            xs, sids = zip(*batch)
            yield batch_indices, (torch.stack(xs), torch.tensor(sids))

class SubjectDataset(Dataset):
    def __init__(self, data, subject_ids):
        self.data = data
        self.subject_ids = subject_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.subject_ids[idx]

class SubjectPairBatchSampler(Sampler):
    def __init__(self, subject_ids, batch_subjects=16):
        self.subject_to_indices = {}
        for idx, sid in enumerate(subject_ids):
            self.subject_to_indices.setdefault(sid, []).append(idx)
        self.subjects = list(self.subject_to_indices.keys())
        self.batch_subjects = batch_subjects

    def __iter__(self):
        random.shuffle(self.subjects)
        for i in range(0, len(self.subjects), self.batch_subjects):
            selected_subjects = self.subjects[i:i+self.batch_subjects]
            batch_indices = []
            for sid in selected_subjects:
                indices = self.subject_to_indices[sid]
                if len(indices) >= 2:
                    pair = random.sample(indices, 2)
                    batch_indices.extend(pair)
            if len(batch_indices) >= 2:
                yield batch_indices

    def __len__(self):
        return len(self.subjects) // self.batch_subjects


def contrastive_loss(x, subject_ids, temperature=0.1):
    B, D = x.size()
    x = F.normalize(x, dim=1)
    sim_matrix = torch.matmul(x, x.T) / temperature  # [B, B]

    mask = subject_ids.unsqueeze(1) == subject_ids.unsqueeze(0)  # [B, B]
    pos_mask = mask.fill_diagonal_(False)
    neg_mask = ~mask

    loss = []
    for i in range(B):
        pos_sim = sim_matrix[i][pos_mask[i]]
        neg_sim = sim_matrix[i][neg_mask[i]]

        if len(pos_sim) == 0:
            continue

        pos_term = torch.logsumexp(pos_sim, dim=0)
        neg_term = torch.logsumexp(neg_sim, dim=0)
        loss_i = - (pos_term - neg_term)
        loss.append(loss_i)

    return torch.stack(loss).mean()


def train(model, dataloader, wid2index=None, epochs=100, lr=1e-3, lambda_reg=1e-3, device='cuda', log_path=None):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    index2wid = {v: k for k, v in wid2index.items()}
    # 用于记录每个样本在每一轮训练中的结构图 A
    perturbation_log = {}  # {index: [A_epoch0, A_epoch1, ...]}

    model.train()
    loss_log = []
    for epoch in range(epochs):
        total_loss = 0
        for batch_indices, (x, subject_ids) in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = x.to(device)
            subject_ids = subject_ids.to(device)

            x_embed, A = model(x)  # x_embed: [B, N, D], A: [B, N, N]

            loss_contrast = contrastive_loss(x_embed, subject_ids)
            loss_reg = lambda_reg * A.abs().mean()
            loss = loss_contrast + loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 记录结构图（detach 防止记录图结构）
            for i, (a_i, sid) in enumerate(zip(A.detach().cpu(), subject_ids.cpu().tolist())):
                idx = batch_indices[i]
                if index2wid[idx] not in perturbation_log:
                    perturbation_log[index2wid[idx]] = []
                perturbation_log[index2wid[idx]].append(a_i)

        print(f"[Epoch {epoch+1}] Total Loss = {total_loss / len(dataloader):.4f}")
        loss_log.append(total_loss / len(dataloader))

    # 保存扰动日志
    if log_path:
        torch.save(perturbation_log, log_path)
        print(f"Perturbation log saved to {log_path}")
    np.savetxt('./loss.txt', np.array(loss_log))
    return 0


if __name__ == "__main__":
    # num_samples = 200
    # num_nodes = 33
    # dim = 64
    #TODO: 训练完用analyze_and_select.py做coreset selection

    os.makedirs('MH_Coresets_ours_list', exist_ok=True)


    tasks = ['sub']
    repeats = [1]#, 2, 3]

    for out_dim in [32]:#, 32, 64, 128, 256, 'data_dim']:
        for head_num in [16]:#, 4, 8, 16, 32]:
            for task in tasks:
                task = task + 'C'
                for repeat in repeats:
                    print(f"task: {task}, repeat: {repeat}")
                    print('-' * 60)
                    log_file = f"./log_{task}_r{repeat}_{head_num}, {out_dim}.pt"
                    data_set = TSWdataset(task=task)
                    data = []
                    subject_ids = []
                    for sample in data_set.samples:
                        data.append(sample[0].squeeze())
                        subject_ids.append(data_set.sub2idx[sample[3].split('_')[0]])
                    data = torch.tensor(data, dtype=torch.float32)
                    num_samples, num_nodes, dim = data.shape[0], data.shape[1], data.shape[2]
                    print(f"num_samples: {num_samples}, num_nodes: {num_nodes}, dim: {dim}")
                    suject_ids = torch.tensor(subject_ids, dtype=torch.int64)

                    dataset = SubjectDataset(data, subject_ids)
                    sampler = SubjectPairBatchSampler(subject_ids, batch_subjects=64)
                    dataloader = IndexDataLoader(dataset, batch_sampler=sampler)
                    if out_dim == 'data_dim':
                        model = StructureModel(in_dim=dim, out_dim=dim, num_heads=head_num)
                    else:
                        model = StructureModel(in_dim=dim, out_dim=out_dim, num_heads=head_num)
                    stat = train(model, dataloader, data_set.wid2index, epochs=1000, lr=1e-3, lambda_reg=1e-4, log_path=log_file)
