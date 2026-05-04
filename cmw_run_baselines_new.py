#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


INPUT_DIR = Path("cmw")
N_SEEDS = 10
TINYBERT_MODEL = "huawei-noah/TinyBERT_General_4L_312D"
TINYBERT_CACHE = INPUT_DIR / "tinybert_embeddings_new.npy"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_new_annotated_data(input_dir: Path):
    nodes = pd.read_csv(input_dir / "nodes_annotated_new.csv")
    comments = pd.read_csv(input_dir / "comments_with_stances_new.csv")
    ann = pd.read_csv(input_dir / "annotation_template_annotated_new.csv")

    if "stance" not in nodes.columns:
        nodes = nodes.merge(ann[["user_id", "stance"]], on="user_id", how="left")
    else:
        ann_small = ann[["user_id", "stance"]].rename(columns={"stance": "stance_ann"})
        nodes = nodes.merge(ann_small, on="user_id", how="left")
        nodes["stance"] = nodes["stance_ann"].fillna(nodes["stance"])
        nodes = nodes.drop(columns=["stance_ann"])

    nodes = nodes.dropna(subset=["user_id", "stance"]).copy()
    nodes["user_id"] = nodes["user_id"].astype(str)
    nodes["all_text"] = nodes["all_text"].fillna("").astype(str)
    nodes["stance"] = nodes["stance"].astype(str)
    nodes["stance_binary"] = np.where(nodes["stance"] == "pro_op", "pro_op", "not_pro_op")

    comments = comments.copy()
    comments["author"] = comments["author"].astype(str)
    comments["parent_author"] = comments["parent_author"].astype(str)

    return nodes, comments


def build_graph(nodes: pd.DataFrame, comments: pd.DataFrame):
    user_ids = nodes["user_id"].tolist()
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    n = len(user_ids)

    G = nx.DiGraph()
    for _, row in nodes.iterrows():
        G.add_node(row["user_id"], stance=row["stance"], text=row["all_text"])

    edge_df = comments[["author", "parent_author"]].dropna()
    edge_df = edge_df[
        (edge_df["author"] != "")
        & (edge_df["parent_author"] != "")
        & (edge_df["author"] != "None")
        & (edge_df["parent_author"] != "None")
        & (edge_df["author"] != "nan")
        & (edge_df["parent_author"] != "nan")
    ]

    grouped = edge_df.groupby(["author", "parent_author"]).size().reset_index(name="weight")
    for _, row in grouped.iterrows():
        src = row["author"]
        dst = row["parent_author"]
        if src in user_to_idx and dst in user_to_idx:
            G.add_edge(src, dst, weight=int(row["weight"]))

    A = np.zeros((n, n), dtype=np.float32)
    for u, v, data in G.edges(data=True):
        i = user_to_idx[u]
        j = user_to_idx[v]
        w = float(data.get("weight", 1.0))
        A[i, j] += w
        A[j, i] += w

    A += np.eye(n, dtype=np.float32)
    deg = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(deg, 1e-8, None))).astype(np.float32)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return G, A_norm


def build_text_features(nodes: pd.DataFrame, max_features: int = 1500, svd_dim: int = 64):
    texts = nodes["all_text"].fillna("").astype(str).tolist()
    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=1,
    )
    X_sparse = tfidf.fit_transform(texts)
    k = min(svd_dim, X_sparse.shape[0] - 1, X_sparse.shape[1] - 1)
    if k >= 2:
        svd = TruncatedSVD(n_components=k, random_state=0)
        X = svd.fit_transform(X_sparse).astype(np.float32)
    else:
        X = X_sparse.toarray().astype(np.float32)
    return X


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def build_tinybert_embeddings(
    texts: list[str],
    model_name: str = TINYBERT_MODEL,
    batch_size: int = 16,
    max_length: int = 256,
    device: str | None = None,
) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.eval()
    mdl.to(device)

    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = mdl(**enc)
            emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
            outs.append(emb.cpu().numpy().astype(np.float32))

    return np.vstack(outs)


def load_or_build_tinybert_embeddings(nodes: pd.DataFrame, cache_path: Path = TINYBERT_CACHE) -> np.ndarray:
    if cache_path.exists():
        X = np.load(cache_path)
        if X.shape[0] == len(nodes):
            return X.astype(np.float32)

    texts = nodes["all_text"].fillna("").astype(str).tolist()
    X = build_tinybert_embeddings(texts)
    np.save(cache_path, X)
    return X


def make_splits(y: np.ndarray, seed: int):
    idx = np.arange(len(y))
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx, y, test_size=0.4, random_state=seed, stratify=y
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    return train_idx, val_idx, test_idx


def metrics_dict(y_true, y_pred):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def oversample_training_data(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    y_train = y[train_idx]
    counts = Counter(y_train)
    max_count = max(counts.values())
    sampled_idx = []
    for cls in sorted(counts):
        cls_idx = train_idx[y_train == cls]
        draws = rng.choice(cls_idx, size=max_count, replace=True)
        sampled_idx.append(draws)
    sampled_idx = np.concatenate(sampled_idx)
    rng.shuffle(sampled_idx)
    return X[sampled_idx], y[sampled_idx]


def class_weights_from_train(y: np.ndarray, train_idx: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y[train_idx], minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.clip(counts, 1.0, None)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def split_count_rows(y: np.ndarray, idxs: np.ndarray, split_name: str, seed: int, id_to_label: dict[int, str]):
    counts = Counter(y[idxs])
    rows = []
    for cls_id, cls_name in id_to_label.items():
        rows.append({"seed": seed, "split": split_name, "label": cls_name, "count": int(counts.get(cls_id, 0))})
    return rows


def run_majority(y, train_idx, test_idx):
    majority = Counter(y[train_idx]).most_common(1)[0][0]
    pred = np.full(len(test_idx), majority)
    return pred, metrics_dict(y[test_idx], pred)


def run_text_logreg(X, y, train_idx, test_idx, seed):
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx])
    return pred, metrics_dict(y[test_idx], pred)


def run_text_mlp(X, y, train_idx, test_idx, seed):
    X_train_bal, y_train_bal = oversample_training_data(X, y, train_idx, seed)
    clf = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=1000,
        random_state=seed,
    )
    clf.fit(X_train_bal, y_train_bal)
    pred = clf.predict(X[test_idx])
    return pred, metrics_dict(y[test_idx], pred)


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = adj @ x
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = adj @ x
        x = self.fc2(x)
        return x


def run_gcn(X, A_norm, y, train_idx, val_idx, test_idx, seed):
    set_seed(seed)
    x = torch.tensor(X, dtype=torch.float32)
    adj = torch.tensor(A_norm, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)

    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    model = GCN(
        in_dim=X.shape[1],
        hidden_dim=32,
        out_dim=len(np.unique(y)),
        dropout=0.4,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    class_weights = class_weights_from_train(y, train_idx, len(np.unique(y)))

    best_state = None
    best_val_f1 = -1.0
    wait = 0
    patience = 50

    for _ in range(300):
        model.train()
        opt.zero_grad()
        logits = model(x, adj)
        loss = F.cross_entropy(logits[train_mask], yt[train_mask], weight=class_weights)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x, adj)[val_mask]
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
            val_f1 = f1_score(y[val_idx], val_pred, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_pred = model(x, adj)[test_mask].argmax(dim=1).cpu().numpy()

    return test_pred, metrics_dict(y[test_idx], test_pred)


def build_edge_index_and_adj(G: nx.DiGraph, user_ids: list[str]):
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    n = len(user_ids)
    adj_lists = [set() for _ in range(n)]

    for u, v in G.edges():
        i = user_to_idx[u]
        j = user_to_idx[v]
        adj_lists[i].add(j)
        adj_lists[j].add(i)

    for i in range(n):
        adj_lists[i].add(i)

    return [sorted(neigh) for neigh in adj_lists]


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj_lists: list[list[int]]) -> torch.Tensor:
        device = x.device
        neigh_means = []
        for neigh in adj_lists:
            idx = torch.tensor(neigh, dtype=torch.long, device=device)
            neigh_means.append(x.index_select(0, idx).mean(dim=0))
        neigh_mean = torch.stack(neigh_means, dim=0)
        out = torch.cat([x, neigh_mean], dim=1)
        out = self.lin(out)
        return out


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.4):
        super().__init__()
        self.sage1 = GraphSAGELayer(in_dim, hidden_dim, dropout=dropout)
        self.sage2 = GraphSAGELayer(hidden_dim, out_dim, dropout=dropout)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj_lists: list[list[int]]) -> torch.Tensor:
        x = self.sage1(x, adj_lists)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage2(x, adj_lists)
        return x


def run_graphsage(X, G, user_ids, y, train_idx, val_idx, test_idx, seed):
    set_seed(seed)
    x = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    adj_lists = build_edge_index_and_adj(G, user_ids)

    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    model = GraphSAGE(
        in_dim=X.shape[1],
        hidden_dim=64,
        out_dim=len(np.unique(y)),
        dropout=0.4,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    class_weights = class_weights_from_train(y, train_idx, len(np.unique(y)))

    best_state = None
    best_val_f1 = -1.0
    wait = 0
    patience = 50

    for _ in range(300):
        model.train()
        opt.zero_grad()
        logits = model(x, adj_lists)
        loss = F.cross_entropy(logits[train_mask], yt[train_mask], weight=class_weights)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x, adj_lists)[val_mask].argmax(dim=1).cpu().numpy()
            val_f1 = f1_score(y[val_idx], val_pred, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_pred = model(x, adj_lists)[test_mask].argmax(dim=1).cpu().numpy()

    return test_pred, metrics_dict(y[test_idx], test_pred)


def summarize(results_df: pd.DataFrame):
    return (
        results_df.groupby("model")[["macro_f1", "accuracy", "balanced_accuracy"]]
        .agg(["mean", "std"])
        .round(4)
    )


def run_experiment(nodes_df: pd.DataFrame, comments_df: pd.DataFrame, label_col: str, task_name: str):
    work_nodes = nodes_df.copy()
    work_nodes["stance"] = work_nodes[label_col].astype(str)

    G, A_norm = build_graph(work_nodes, comments_df)
    X = build_text_features(work_nodes)
    X_tinybert = load_or_build_tinybert_embeddings(work_nodes)
    user_ids = work_nodes["user_id"].tolist()

    label_names = sorted(work_nodes[label_col].unique())
    label_to_id = {label: i for i, label in enumerate(label_names)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    y = np.array([label_to_id[s] for s in work_nodes[label_col]], dtype=np.int64)

    rows = []
    split_rows = []

    for seed in range(N_SEEDS):
        train_idx, val_idx, test_idx = make_splits(y, seed)
        split_rows.extend(split_count_rows(y, train_idx, "train", seed, id_to_label))
        split_rows.extend(split_count_rows(y, val_idx, "val", seed, id_to_label))
        split_rows.extend(split_count_rows(y, test_idx, "test", seed, id_to_label))

        for model_name, runner in [
            ("majority", lambda: run_majority(y, train_idx, test_idx)),
            ("text_logreg", lambda: run_text_logreg(X, y, train_idx, test_idx, seed)),
            ("text_mlp_oversampled", lambda: run_text_mlp(X, y, train_idx, test_idx, seed)),
            ("gcn_weighted", lambda: run_gcn(X, A_norm, y, train_idx, val_idx, test_idx, seed)),
            ("graphsage_tinybert", lambda: run_graphsage(X_tinybert, G, user_ids, y, train_idx, val_idx, test_idx, seed)),
        ]:
            _, mets = runner()
            rows.append({"task": task_name, "seed": seed, "model": model_name, **mets})

    return pd.DataFrame(rows), pd.DataFrame(split_rows)


def main():
    for p in (
        INPUT_DIR / "annotation_template_annotated_new.csv",
        INPUT_DIR / "comments_with_stances_new.csv",
        INPUT_DIR / "nodes_annotated_new.csv",
    ):
        if not p.exists():
            raise FileNotFoundError(p)

    nodes_new, comments_new = load_new_annotated_data(INPUT_DIR)

    print("Users (labeled nodes):", len(nodes_new))
    print("3-class label counts:")
    print(nodes_new["stance"].value_counts())
    print("\nBinary label counts:")
    print(nodes_new["stance_binary"].value_counts())

    results_threeway, splits_threeway = run_experiment(
        nodes_new,
        comments_new,
        label_col="stance",
        task_name="cmw_new_threeway",
    )
    print("\n3-class split counts:")
    print(
        splits_threeway.pivot_table(
            index=["seed", "split"], columns="label", values="count", fill_value=0
        )
    )
    print("\n3-class summary:")
    print(summarize(results_threeway))

    results_binary, splits_binary = run_experiment(
        nodes_new,
        comments_new,
        label_col="stance_binary",
        task_name="cmw_new_pro_vs_not_pro",
    )
    print("\nBinary split counts:")
    print(
        splits_binary.pivot_table(
            index=["seed", "split"], columns="label", values="count", fill_value=0
        )
    )
    print("\nBinary summary:")
    print(summarize(results_binary))

    results_all = pd.concat([results_threeway, results_binary], ignore_index=True)
    split_counts_all = pd.concat(
        [
            splits_threeway.assign(task="cmw_new_threeway"),
            splits_binary.assign(task="cmw_new_pro_vs_not_pro"),
        ],
        ignore_index=True,
    )

    results_path = INPUT_DIR / "baseline_results_new.csv"
    summary_path = INPUT_DIR / "baseline_results_new_summary.csv"
    split_counts_path = INPUT_DIR / "baseline_split_counts_new.csv"

    results_all.to_csv(results_path, index=False)
    (
        results_all.groupby(["task", "model"])[["macro_f1", "accuracy", "balanced_accuracy"]]
        .agg(["mean", "std"])
        .round(4)
        .to_csv(summary_path)
    )
    split_counts_all.to_csv(split_counts_path, index=False)

    print(f"\nWrote {results_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {split_counts_path}")


if __name__ == "__main__":
    main()
