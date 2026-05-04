#!/usr/bin/env python3
"""
Fetch one public Reddit thread via the .json endpoint and generate:
  1) comments.csv
  2) nodes.csv
  3) annotation_template.csv

No PRAW, no Reddit app credentials required.

Target thread:
https://www.reddit.com/r/changemyview/comments/p4gxiv/cmv_the_abortion_debate_has_no_resolution_since/

Usage:
  pip install requests pandas
  python reddit_thread_json_to_csv.py --out_dir reddit_cmv_abortion

Outputs:
  - comments.csv
  - nodes.csv
  - annotation_template.csv
"""

import os
import time
import argparse
from collections import defaultdict

import requests
import pandas as pd


THREAD_JSON_URL = (
    "https://www.reddit.com/r/changemyview/comments/p4gxiv/"
    "cmv_the_abortion_debate_has_no_resolution_since/.json"
)

HEADERS = {
    "User-Agent": "stance-benchmark-script/0.1"
}


def normalize_author(author):
    if author is None:
        return None
    author = str(author).strip()
    if not author or author.lower() in {"[deleted]", "deleted"}:
        return None
    return author


def extract_comment_rows(comment_node, submission_meta, rows):
    """
    Recursively walk Reddit JSON comments.

    comment_node is expected to be a dict with:
      kind: "t1"
      data: {...}
    """
    if not isinstance(comment_node, dict):
        return

    kind = comment_node.get("kind")
    data = comment_node.get("data", {})

    if kind != "t1":
        return

    author = normalize_author(data.get("author"))
    if author is None:
        # still recurse into replies if present
        replies = data.get("replies")
        if isinstance(replies, dict):
            children = replies.get("data", {}).get("children", [])
            for child in children:
                extract_comment_rows(child, submission_meta, rows)
        return

    parent_id_full = data.get("parent_id")
    parent_comment_id = None
    if isinstance(parent_id_full, str) and parent_id_full.startswith("t1_"):
        parent_comment_id = parent_id_full[3:]

    row = {
        "submission_id": submission_meta["submission_id"],
        "submission_title": submission_meta["submission_title"],
        "subreddit": submission_meta["subreddit"],
        "comment_id": data.get("id"),
        "author": author,
        "parent_comment_id": parent_comment_id,
        "body": data.get("body", ""),
        "score": data.get("score"),
        "created_utc": data.get("created_utc"),
        "permalink": (
            f"https://www.reddit.com{data.get('permalink')}"
            if data.get("permalink")
            else None
        ),
        "depth": data.get("depth"),
    }
    rows.append(row)

    replies = data.get("replies")
    if isinstance(replies, dict):
        children = replies.get("data", {}).get("children", [])
        for child in children:
            extract_comment_rows(child, submission_meta, rows)


def build_parent_author(comments_df):
    """
    Add parent_author by matching parent_comment_id to comment_id.
    """
    id_to_author = dict(zip(comments_df["comment_id"], comments_df["author"]))
    comments_df = comments_df.copy()
    comments_df["parent_author"] = comments_df["parent_comment_id"].map(id_to_author)
    return comments_df


def build_nodes(comments_df):
    grouped = defaultdict(list)

    for _, row in comments_df.iterrows():
        grouped[row["author"]].append(str(row["body"]))

    node_rows = []
    for user_id, texts in grouped.items():
        node_rows.append(
            {
                "user_id": user_id,
                "num_comments": len(texts),
                "all_text": "\n\n".join(texts).strip(),
            }
        )

    nodes_df = pd.DataFrame(node_rows)
    if not nodes_df.empty:
        nodes_df = nodes_df.sort_values(
            by=["num_comments", "user_id"], ascending=[False, True]
        )
    return nodes_df


def build_annotation_template(nodes_df):
    ann_df = pd.DataFrame(
        {
            "user_id": nodes_df["user_id"],
            "stance": "",
            "notes": "",
        }
    )
    return ann_df.sort_values(by="user_id")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default="reddit_cmv_abortion",
        help="Directory to save the CSV files",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Optional sleep before request",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.sleep > 0:
        time.sleep(args.sleep)

    resp = requests.get(THREAD_JSON_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError("Unexpected Reddit JSON structure.")

    post_listing = payload[0]
    comments_listing = payload[1]

    post_children = post_listing.get("data", {}).get("children", [])
    if not post_children:
        raise RuntimeError("Could not find submission metadata.")

    post_data = post_children[0].get("data", {})
    submission_meta = {
        "submission_id": post_data.get("id"),
        "submission_title": post_data.get("title"),
        "subreddit": post_data.get("subreddit"),
    }

    rows = []
    comment_children = comments_listing.get("data", {}).get("children", [])
    for child in comment_children:
        extract_comment_rows(child, submission_meta, rows)

    comments_df = pd.DataFrame(rows)
    if comments_df.empty:
        raise RuntimeError("No comments were extracted.")

    comments_df = build_parent_author(comments_df)

    nodes_df = build_nodes(comments_df)
    ann_df = build_annotation_template(nodes_df)

    comments_path = os.path.join(args.out_dir, "comments.csv")
    nodes_path = os.path.join(args.out_dir, "nodes.csv")
    ann_path = os.path.join(args.out_dir, "annotation_template.csv")

    comments_df.to_csv(comments_path, index=False)
    nodes_df.to_csv(nodes_path, index=False)
    ann_df.to_csv(ann_path, index=False)

    print(f"Wrote {comments_path}")
    print(f"Wrote {nodes_path}")
    print(f"Wrote {ann_path}")
    print()
    print(f"Comments: {len(comments_df)}")
    print(f"Users:    {len(nodes_df)}")
    print()
    print("Next step:")
    print("1. Open nodes.csv while reading user text")
    print("2. Fill stance + notes in annotation_template.csv")
    print("3. Later merge by user_id to build the labeled graph")


if __name__ == "__main__":
    main()
