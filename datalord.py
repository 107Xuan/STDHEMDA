import pandas as pd
import csv
from typing import List, Tuple, Generator
import numpy as np
import torch
from sklearn.model_selection import KFold


def extract_edge_features(data: torch.Tensor, em: torch.Tensor, ed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    edgeData = data.t()
    m_index = edgeData[0].to(em.device)
    d_index = edgeData[1].to(ed.device)
    Em = torch.index_select(em, 0, m_index)
    Ed = torch.index_select(ed, 0, d_index)
    return Em, Ed


def generate_negative_samples(
        num_samples: int,
        num_m: int,
        num_d: int,
        forbidden_set: set,
        rng: np.random.RandomState
) -> List[Tuple[int, int]]:
    neg_pairs = []
    max_possible = num_m * num_d
    forbidden_count = len(forbidden_set)

    if forbidden_count / max_possible > 0.5:
        all_pairs = [(i, j) for i in range(num_m) for j in range(num_d)]
        available_pairs = [p for p in all_pairs if p not in forbidden_set]
        if len(available_pairs) < num_samples:
            raise ValueError(f"Not enough negative samples available: {len(available_pairs)} < {num_samples}")
        selected_indices = rng.choice(len(available_pairs), size=num_samples, replace=False)
        neg_pairs = [available_pairs[i] for i in selected_indices]
    else:
        attempts = 0
        max_attempts = num_samples * 100
        while len(neg_pairs) < num_samples and attempts < max_attempts:
            m_idx = rng.randint(0, num_m)
            d_idx = rng.randint(0, num_d)
            if (m_idx, d_idx) not in forbidden_set:
                neg_pairs.append((m_idx, d_idx))
                forbidden_set.add((m_idx, d_idx))
            attempts += 1

        if len(neg_pairs) < num_samples:
            raise RuntimeError(f"Failed to generate enough negative samples after {max_attempts} attempts")

    return neg_pairs


def load_train_test_rigorous(
        link_file: str,
        mirnas: List[str],
        drugs: List[str],
        n_splits: int = 5,
        random_state: int = 42
) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
    mirna2idx = {n: i for i, n in enumerate(mirnas)}
    drug2idx = {n: i for i, n in enumerate(drugs)}

    all_pos_pairs = []
    try:
        with open(link_file, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < 2:
                    continue
                m, d = row[0], row[1]
                if m in mirna2idx and d in drug2idx:
                    all_pos_pairs.append((mirna2idx[m], drug2idx[d]))
    except FileNotFoundError:
        raise FileNotFoundError(f"Link file not found: {link_file}")
    except Exception as e:
        raise RuntimeError(f"Error reading link file: {str(e)}")

    if not all_pos_pairs:
        raise ValueError("No valid positive pairs found in the link file")

    all_pos_pairs = np.array(all_pos_pairs, dtype=np.int64)
    all_pos_set = set(map(tuple, all_pos_pairs))

    num_m, num_d = len(mirnas), len(drugs)
    rng = np.random.RandomState(random_state)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_idx, (train_pos_idx, val_pos_idx) in enumerate(kf.split(all_pos_pairs)):
        train_pos_pairs = all_pos_pairs[train_pos_idx]
        val_pos_pairs = all_pos_pairs[val_pos_idx]

        train_rng = np.random.RandomState(random_state + fold_idx * 2)
        val_rng = np.random.RandomState(random_state + fold_idx * 2 + 1)

        train_forbidden_set = all_pos_set.copy()
        train_neg_pairs = generate_negative_samples(
            num_samples=len(train_pos_pairs),
            num_m=num_m,
            num_d=num_d,
            forbidden_set=train_forbidden_set,
            rng=train_rng
        )

        val_forbidden_set = all_pos_set.copy()
        val_neg_pairs = generate_negative_samples(
            num_samples=len(val_pos_pairs),
            num_m=num_m,
            num_d=num_d,
            forbidden_set=val_forbidden_set,
            rng=val_rng
        )

        train_pairs = np.vstack([train_pos_pairs, np.array(train_neg_pairs, dtype=np.int64)])
        train_labels = np.hstack([np.ones(len(train_pos_pairs)), np.zeros(len(train_neg_pairs))])

        val_pairs = np.vstack([val_pos_pairs, np.array(val_neg_pairs, dtype=np.int64)])
        val_labels = np.hstack([np.ones(len(val_pos_pairs)), np.zeros(len(val_neg_pairs))])

        train_perm = rng.permutation(len(train_pairs))
        train_pairs = train_pairs[train_perm]
        train_labels = train_labels[train_perm]

        val_perm = rng.permutation(len(val_pairs))
        val_pairs = val_pairs[val_perm]
        val_labels = val_labels[val_perm]

        train_pairs[:, 1] += num_m
        val_pairs[:, 1] += num_m

        yield (
            torch.from_numpy(train_pairs).long(),
            torch.from_numpy(train_labels).float(),
            torch.from_numpy(val_pairs).long(),
            torch.from_numpy(val_labels).float()
        )


def load_sim_edge_index(
        sim_file: str,
        sim_threshold: float = 0.5,
        include_self_loops: bool = False
) -> torch.Tensor:
    if not 0 <= sim_threshold <= 1:
        raise ValueError("sim_threshold must be between 0 and 1")

    try:
        sim_df = pd.read_csv(sim_file, index_col=0)
    except FileNotFoundError:
        raise FileNotFoundError(f"Similarity file not found: {sim_file}")
    except Exception as e:
        raise RuntimeError(f"Error reading similarity file: {str(e)}")

    if sim_df.shape[0] != sim_df.shape[1]:
        raise ValueError(f"Similarity matrix must be square, got shape {sim_df.shape}")

    n = len(sim_df)
    if n == 0:
        return torch.empty((2, 0), dtype=torch.long)

    sim_matrix = sim_df.values
    edges = []

    for i in range(n):
        start_j = i if include_self_loops else i + 1
        for j in range(start_j, n):
            sim_val = sim_matrix[i, j]
            if np.isnan(sim_val):
                continue
            if sim_val > sim_threshold:
                if i == j:
                    edges.append((i, j))
                else:
                    edges.extend([(i, j), (j, i)])

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
    return edge_index


def load_edge_ws_index(
        train_pairs: torch.Tensor,
        train_labels: torch.Tensor,
        num_mirna: int
) -> torch.Tensor:
    train_pairs = torch.as_tensor(train_pairs)
    train_labels = torch.as_tensor(train_labels)

    train_pos = train_pairs[train_labels == 1]

    mirna_indices = train_pos[:, 0]
    drug_global_indices = train_pos[:, 1]
    drug_local_indices = drug_global_indices - num_mirna

    if mirna_indices.numel() > 0:
        assert mirna_indices.min() >= 0, f"Invalid miRNA index: {mirna_indices.min()}"
        assert drug_local_indices.min() >= 0, f"Invalid drug local index: {drug_local_indices.min()}"

        edge_index = torch.stack([mirna_indices, drug_local_indices], dim=0).long()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return edge_index