#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(input_dir: Path):
    nodes_path = input_dir / "nodes_annotated.csv"
    comments_path = input_dir / "comments_with_stances.csv"
    ann_path = input_dir / "annotation_template_annotated.csv"

    if not nodes_path.exists():
        raise FileNotFoundError(nodes_path)
    if not comments_path.exists():
        raise FileNotFoundError(comments_path)
    if not ann_path.exists():
        raise FileNotFoundError(ann_path)

    nodes = pd.read_csv(nodes_path)
    comments = pd.read_csv(comments_path)
    ann = pd.read_csv(ann_path)

    # Prefer node table as main source, but refresh stance from annotation file if available
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
    ]

    # user -> parent_user
    grouped = edge_df.groupby(["author", "parent_author"]).size().reset_index(name="weight")
    for _, row in grouped.iterrows():
        src = row["author"]
        dst = row["parent_author"]
        if src in user_to_idx and dst in user_to_idx:
            G.add_edge(src, dst, weight=int(row["weight"]))

    # Build symmetrized normalized adjacency for vanilla GCN
    A = np.zeros((n, n), dtype=np.float32)
    for u, v, data in G.edges(data=True):
        i = user_to_idx[u]
        j = user_to_idx[v]
        w = float(data.get("weight", 1.0))
        A[i, j] += w
        A[j, i] += w  # symmetrize for basic GCN

    A += np.eye(n, dtype=np.float32)
    deg = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(deg, 1e-8, None))).astype(np.float32)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return G, user_ids, user_to_idx, A_norm


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
    }


def run_majority(y, train_idx, test_idx):
    majority = Counter(y[train_idx]).most_common(1)[0][0]
    pred = np.full(len(test_idx), majority)
    return pred, metrics_dict(y[test_idx], pred)


def run_text_logreg(X, y, train_idx, test_idx, seed):
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx])
    return pred, metrics_dict(y[test_idx], pred)


def run_text_mlp(X, y, train_idx, test_idx, seed):
    clf = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=1000,
        random_state=seed,
    )
    clf.fit(X[train_idx], y[train_idx])
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

    best_state = None
    best_val_f1 = -1.0
    patience = 50
    wait = 0

    for _ in range(300):
        model.train()
        opt.zero_grad()
        logits = model(x, adj)
        loss = F.cross_entropy(logits[train_mask], yt[train_mask])
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


def summarize(results_df: pd.DataFrame):
    return (
        results_df.groupby("model")[["macro_f1", "accuracy"]]
        .agg(["mean", "std"])
        .round(4)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=".")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--output_prefix", type=str, default="baseline_results")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    nodes, comments = load_data(input_dir)

    label_names = sorted(nodes["stance"].unique())
    label_to_id = {label: i for i, label in enumerate(label_names)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    y = np.array([label_to_id[s] for s in nodes["stance"]], dtype=np.int64)
    X = build_text_features(nodes)
    _, user_ids, _, A_norm = build_graph(nodes, comments)

    print("Users:", len(nodes))
    print("Class counts:")
    print(nodes["stance"].value_counts())
    print()
    print("Using repeated stratified 60/20/20 splits.")
    print()

    all_rows = []
    last_split_preds = []

    for seed in range(args.n_seeds):
        train_idx, val_idx, test_idx = make_splits(y, seed)

        models = {
            "majority": lambda: run_majority(y, train_idx, test_idx),
            "text_logreg": lambda: run_text_logreg(X, y, train_idx, test_idx, seed),
            "text_mlp": lambda: run_text_mlp(X, y, train_idx, test_idx, seed),
            "gcn": lambda: run_gcn(X, A_norm, y, train_idx, val_idx, test_idx, seed),
        }

        for model_name, runner in models.items():
            pred, mets = runner()
            row = {
                "seed": seed,
                "model": model_name,
                "macro_f1": mets["macro_f1"],
                "accuracy": mets["accuracy"],
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
            }
            all_rows.append(row)

            if seed == args.n_seeds - 1:
                for idx, p in zip(test_idx, pred):
                    last_split_preds.append({
                        "seed": seed,
                        "model": model_name,
                        "user_id": user_ids[idx],
                        "true_label": id_to_label[y[idx]],
                        "pred_label": id_to_label[int(p)],
                        "split": "test",
                    })

    results_df = pd.DataFrame(all_rows)
    summary_df = summarize(results_df)
    preds_df = pd.DataFrame(last_split_preds)

    results_path = input_dir / f"{args.output_prefix}.csv"
    summary_path = input_dir / f"{args.output_prefix}_summary.csv"
    preds_path = input_dir / f"{args.output_prefix}_last_seed_test_predictions.csv"
    meta_path = input_dir / f"{args.output_prefix}_meta.json"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path)
    preds_df.to_csv(preds_path, index=False)

    meta = {
        "label_to_id": label_to_id,
        "n_users": int(len(nodes)),
        "class_counts": nodes["stance"].value_counts().to_dict(),
        "split_scheme": "Repeated stratified 60/20/20",
        "n_seeds": args.n_seeds,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(results_df.round(4).to_string(index=False))
    print()
    print("Summary:")
    print(summary_df)
    print()
    print("Saved:")
    print(results_path)
    print(summary_path)
    print(preds_path)
    print(meta_path)


if __name__ == "__main__":
    main()
