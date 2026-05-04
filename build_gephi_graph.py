#!/usr/bin/env python3
"""
Build a directed NetworkX graph from annotated Reddit CSV files
and export it for Gephi.

Expected files in input_dir:
  - comments_with_stances.csv
  - nodes_annotated.csv
  - annotation_template_annotated.csv

Outputs:
  - reddit_graph.gexf
  - reddit_graph.graphml

Usage:
  python build_gephi_graph.py --input_dir .
  python build_gephi_graph.py --input_dir /path/to/folder --output_prefix cmv_abortion
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import networkx as nx


def clean_value(x):
    """Convert NaN-like pandas values to None; stringify others when needed."""
    if pd.isna(x):
        return None
    return x


def safe_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def build_graph(
    comments_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    ann_df: pd.DataFrame,
) -> nx.DiGraph:
    """
    Build a directed graph:
      node = user
      edge u -> v means user u replied to user v
    """
    required_comment_cols = {"author", "parent_author"}
    missing_comment = required_comment_cols - set(comments_df.columns)
    if missing_comment:
        raise ValueError(
            f"comments_with_stances.csv missing columns: {sorted(missing_comment)}"
        )

    required_node_cols = {"user_id"}
    missing_nodes = required_node_cols - set(nodes_df.columns)
    if missing_nodes:
        raise ValueError(f"nodes_annotated.csv missing columns: {sorted(missing_nodes)}")

    required_ann_cols = {"user_id", "stance"}
    missing_ann = required_ann_cols - set(ann_df.columns)
    if missing_ann:
        raise ValueError(
            f"annotation_template_annotated.csv missing columns: {sorted(missing_ann)}"
        )

    # Normalize key columns
    comments_df = comments_df.copy()
    nodes_df = nodes_df.copy()
    ann_df = ann_df.copy()

    comments_df["author"] = comments_df["author"].astype(str)
    comments_df["parent_author"] = comments_df["parent_author"].astype(str)
    nodes_df["user_id"] = nodes_df["user_id"].astype(str)
    ann_df["user_id"] = ann_df["user_id"].astype(str)

    # Merge node info + annotation info on user_id
    merged_nodes = nodes_df.merge(
        ann_df,
        on="user_id",
        how="left",
        suffixes=("", "_ann"),
    )

    G = nx.DiGraph()

    # Add nodes with attributes
    for _, row in merged_nodes.iterrows():
        user_id = row["user_id"]

        attrs = {
            "label": user_id,  # handy for Gephi labels
            "stance": safe_text(row["stance"]) if "stance" in row else "",
            "notes": safe_text(row["notes"]) if "notes" in row else "",
        }

        if "num_comments" in row:
            attrs["num_comments"] = clean_value(row["num_comments"])

        if "all_text" in row:
            # Gephi can handle text attrs, though huge text can make files bulky
            attrs["all_text"] = safe_text(row["all_text"])

        G.add_node(user_id, **attrs)

    # Count repeated reply interactions into weighted directed edges
    edge_weights = (
        comments_df[
            comments_df["author"].notna()
            & comments_df["parent_author"].notna()
            & (comments_df["author"] != "None")
            & (comments_df["parent_author"] != "None")
            & (comments_df["author"] != "")
            & (comments_df["parent_author"] != "")
        ]
        .groupby(["author", "parent_author"])
        .size()
        .reset_index(name="weight")
    )

    # Add edges
    for _, row in edge_weights.iterrows():
        src = row["author"]
        dst = row["parent_author"]
        weight = int(row["weight"])

        # Ensure nodes exist even if somehow missing from node table
        if src not in G:
            G.add_node(src, label=src, stance="")
        if dst not in G:
            G.add_node(dst, label=dst, stance="")

        G.add_edge(src, dst, weight=weight)

    # Add a few graph-level metadata fields
    G.graph["name"] = "Reddit CMV stance reply graph"
    G.graph["description"] = (
        "Directed user-reply graph from annotated Reddit discussion. "
        "Edge u->v means user u replied to user v."
    )

    return G


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default=".",
        help="Folder containing the three CSV files",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="reddit_graph",
        help="Prefix for exported graph files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    comments_path = input_dir / "comments_with_stances.csv"
    nodes_path = input_dir / "nodes_annotated.csv"
    ann_path = input_dir / "annotation_template_annotated.csv"

    comments_df = read_csv_required(comments_path)
    nodes_df = read_csv_required(nodes_path)
    ann_df = read_csv_required(ann_path)

    G = build_graph(comments_df, nodes_df, ann_df)

    out_gexf = input_dir / f"{args.output_prefix}.gexf"
    out_graphml = input_dir / f"{args.output_prefix}.graphml"

    nx.write_gexf(G, out_gexf)
    nx.write_graphml(G, out_graphml)

    print("Done.")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Wrote: {out_gexf}")
    print(f"Wrote: {out_graphml}")
    print()
    print("In Gephi:")
    print("  1. Open the .gexf file")
    print("  2. Use 'stance' for partition coloring")
    print("  3. Use 'num_comments' or degree for node size")
    print("  4. Use edge 'weight' for edge thickness")


if __name__ == "__main__":
    main()
