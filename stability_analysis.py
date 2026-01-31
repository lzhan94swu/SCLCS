import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter

def plot_smoothed_mean(score_dict):
    # 读取 loss
    with open("./loss.txt", "r") as f:
        loss_values = [float(line.strip()) for line in f if line.strip()]

    # 创建并列子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- 绘制扰动图 ---
    max_len = max(len(v) for v in score_dict.values() if not np.all(v.cpu().numpy() == 0))
    sum_curve = np.zeros(max_len)
    count_curve = np.zeros(max_len)

    for ind, v in enumerate(score_dict.values()):
        seq = v.cpu().numpy()
        if np.all(seq == 0):
            continue
        seq_len = len(seq)
        x = np.arange(1, seq_len + 1)
        if ind < 50:
            ax1.plot(x, seq, color='gray', alpha=0.05, linewidth=0.5)
        sum_curve[:seq_len] += seq
        count_curve[:seq_len] += 1

    valid = count_curve >= 10
    mean_curve = np.zeros_like(sum_curve)
    mean_curve[valid] = sum_curve[valid] / count_curve[valid]
    x_valid = np.arange(1, max_len + 1)[valid]
    y_valid = mean_curve[valid]
    y_smooth = savgol_filter(y_valid, window_length=31, polyorder=3)

    ax1.plot(x_valid, y_smooth, color='blue', linewidth=2, label='SPS')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel(r"$\Delta_e(\mathbf{X})$", fontsize=12)
    ax1.set_title("Perturbation Trends", fontsize=13)
    ax1.set_ylim(0, 4)
    ax1.grid(False)

    # --- 绘制 Loss 曲线 ---
    ax2.plot(np.arange(1, len(loss_values) + 1), loss_values, color='red', linewidth=1)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("Training Loss", fontsize=13)
    ax2.grid(False)

    plt.tight_layout()
    plt.show()

def compute_epochwise_delta(A_seq):
    if len(A_seq) < 2:
        return float('inf')
    A_stack = torch.stack(A_seq, dim=0)
    deltas = A_stack[1:] - A_stack[:-1]
    norms = deltas.view(deltas.size(0), -1).norm(dim=1)
    return norms# .item()


def analyze_and_select(log_path="structure_perturbation_log.pt"):
    perturbation_log = torch.load(log_path)

    score_dict = {}
    for idx, A_seq in tqdm(perturbation_log.items(), desc="Computing epochwise perturbation"):
        score = compute_epochwise_delta(A_seq)
        score_dict[idx] = score

    plot_smoothed_mean(score_dict)


if __name__ == "__main__":
    analyze_and_select(log_path="./log_subC_r1_16, 32.pt")
