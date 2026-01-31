import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import gaussian_kde


def compute_epochwise_delta(A_seq):
    if len(A_seq) < 2:
        return float('inf')
    A_stack = torch.stack(A_seq, dim=0)
    deltas = A_stack[1:] - A_stack[:-1]
    norms = deltas.view(deltas.size(0), -1).norm(dim=1)
    return norms.mean().item()


def select_density_balanced_subset(score_dict, beta=0.05, alpha=0.5, sample_size=None, bandwidth_adjust=1.0):
    indices = list(score_dict.keys())
    scores = np.array([score_dict[i] for i in indices])
    threshold = np.percentile(scores, 100 * (1 - beta))
    kept = [(i, s) for i, s in zip(indices, scores) if s <= threshold]

    if not kept:
        raise ValueError("No samples left after beta filtering.")

    kept_indices, kept_scores = zip(*kept)
    kept_scores = np.array(kept_scores)

    kde = gaussian_kde(kept_scores)
    kde.set_bandwidth(bw_method=kde.factor * bandwidth_adjust)
    densities = kde.evaluate(kept_scores)

    weights = 1.0 / (densities + 1e-8)
    weights /= weights.sum()

    m = sample_size if sample_size is not None else int(len(score_dict) * (1 - alpha))
    selected_indices = np.random.choice(kept_indices, size=m, replace=False, p=weights)
    return selected_indices.tolist()


def analyze_and_select(log_path="structure_perturbation_log.pt",
                       save_score_path="sorted_epochwise_perturbation.csv",
                       save_selected_path="selected_indices.csv",
                       topk=50, beta=0.05, alpha=0.5):
    perturbation_log = torch.load(log_path)

    score_dict = {}
    for idx, A_seq in tqdm(perturbation_log.items(), desc="Computing epochwise perturbation"):
        score = compute_epochwise_delta(A_seq)
        score_dict[idx] = score

    # 打印 top-k 最稳定样本
    sorted_scores = sorted(score_dict.items(), key=lambda x: x[1])
    print(f"\nTop-{topk} most stable samples:")
    for i in range(min(topk, len(sorted_scores))):
        idx, score = sorted_scores[i]
        print(f"Sample {idx}: score={score:.4f}")

    if save_score_path:
        df = pd.DataFrame(sorted_scores, columns=["Sample ID", "SCLCS Score"])
        # torch.save(sorted_scores, save_score_path)
        df.to_csv(save_score_path, index=False)
        print(f"\nSaved sorted scores to: {save_score_path}")

    # 执行密度感知筛选
    selected = select_density_balanced_subset(score_dict, beta=beta, alpha=alpha)
    print(f"\nSelected {len(selected)} core-set samples (density-aware).")

    if save_selected_path:
        torch.save(selected, save_selected_path)
        dens_df = pd.DataFrame({"Sample ID": selected})
        dens_df.to_csv(save_selected_path, index=False)
        print(f"Saved selected indices to: {save_selected_path}")

    return selected


if __name__ == "__main__":
    analyze_and_select(
        log_path="./structure_perturbation_log.pt",
        save_score_path="./sorted_epochwise_perturbation.csv",
        save_selected_path="./selected_indices.csv",
        topk=50,
        beta=0.05,
        alpha=0.5
    )
