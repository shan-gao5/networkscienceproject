# Network science: Reddit CMV stance modeling

Personal project combining **network analysis**, **tabular NLP**, and **node-level classification** on a [r/changemyview](https://www.reddit.com/r/changemyview/) discussion. Users are nodes; directed edges aggregate “who replied to whom.” The goal is to predict **stance labels** (e.g., pro-life / pro-choice / neutral) from text and graph structure under a fixed train/validation/test split.

## What’s in this repo

| Piece | Role |
|--------|------|
| `reddit_thread_json_to_csv.py` | Pulls a public thread via Reddit’s `.json` URL (no API keys) and writes `comments.csv`, `nodes.csv`, and an `annotation_template.csv` for manual or semi-automatic labeling. |
| `build_gephi_graph.py` | Builds a **NetworkX** `DiGraph` from annotated CSVs and exports **Gephi**-friendly `.gexf` / `.graphml`. |
| `run_reddit_baselines.py` | CLI baselines: majority class, TF–IDF + logistic regression / MLP, and a small **GCN** in PyTorch on a symmetrized normalized adjacency; repeated stratified splits and CSV summaries. |
| `reddit_baselines_annotated.ipynb` | End-to-end notebook: load annotated CSVs → NetworkX graph → **TinyBERT** mean-pooled text embeddings per user → same split scheme → baselines plus **GraphSAGE** (PyTorch Geometric). |
| `cmw_jsonl_to_annotation_csvs.ipynb` | Notebook path for turning thread JSONL into annotation-oriented CSVs (workflow helper). |
| `cmw_run_baselines_new.py` | Variant pipeline for a separate `cmw/` annotated layout (binary stance helpers, TinyBERT cache, graph + sklearn + torch models). |

Data produced by the scripts or used for experiments typically lives under folders such as `annotated/`, `reddit_cmv_abortion/`, or `cmw/` (CSVs and optional embedding caches). Exact filenames match what each script expects (see script headers or `--help`).

## Quick start

**1. Fetch raw thread CSVs**

```bash
pip install requests pandas
python reddit_thread_json_to_csv.py --out_dir reddit_cmv_abortion
```

**2. After you have annotated files** (e.g. `nodes_annotated.csv`, `comments_with_stances.csv`, `annotation_template_annotated.csv` in a folder):

```bash
pip install pandas networkx
python build_gephi_graph.py --input_dir annotated
```

**3. Run CLI baselines** (TF–IDF + GCN stack as in `run_reddit_baselines.py`):

```bash
pip install pandas numpy networkx scikit-learn torch
python run_reddit_baselines.py --input_dir annotated --n_seeds 10
```

**4. Notebook pipeline** (TinyBERT + GraphSAGE): open `reddit_baselines_annotated.ipynb` in Jupyter and run top-to-bottom; install `transformers`, `sentencepiece`, and `torch-geometric` as noted in the notebook if those cells are enabled.

## Tech stack

Python, **pandas**, **NetworkX**, **scikit-learn**, **PyTorch**; optional **Hugging Face Transformers** (TinyBERT) and **PyTorch Geometric** (GraphSAGE) in the notebook path.

## Note

Reddit content and rate limits are subject to Reddit’s terms and etiquette. The fetch script uses a simple HTTP `.json` request with a descriptive `User-Agent`; use responsibly and prefer your own saved exports for reproducible papers or interviews.
