# NLU Assignment 2 — Runbook

IITJ web corpus + Word2Vec (Problem 1) and character-level name generation (Problem 2). Run every command from the **repository root** (the folder that contains `problem1/` and `problem2/`).

---

## 0. Environment

**Python:** 3.10+

**Install dependencies** (pick one):

- With **uv** (if `pyproject.toml` is present):

  ```bash
  uv sync
  uv sync --extra problem2    # adds PyTorch, Gemini client, dotenv
  ```

- Otherwise install roughly what `pyproject.toml` lists: `gensim`, `nltk`, `requests`, `beautifulsoup4`, `lxml`, `pdfplumber`, `langdetect`, `matplotlib`, `scikit-learn`, `wordcloud`, and for Problem 2 also `torch`, `google-genai`, `python-dotenv`.

**Problem 2 / Gemini:** set `GOOGLE_API_KEY` or `GEMINI_API_KEY` in the environment or a `.env` file (never commit secrets).

**Crawled PDFs:** `problem1/data/pdfs/*.pdf` may be gitignored. To rebuild them and the corpus, run **Task 1** with network access (or copy PDFs into `problem1/data/pdfs/` yourself).

---

## Problem 1

### Task 1 — Build corpus

Crawls IIT Jodhpur–related URLs, downloads PDFs into `problem1/data/pdfs/`, extracts text, filters English, tokenizes, writes:

- `problem1/output/corpus_tokens.pkl`
- `problem1/output/corpus_meta.json`
- `problem1/output/crawl_log.json`

```bash
uv run python -m problem1.build_corpus --rebuild
```

- First run may download NLTK data (`punkt`, etc.).
- **`--no-crawl`:** only fetch seed URLs + read PDFs already on disk (no BFS).
- Omit **`--rebuild`** if you want to keep an existing `corpus_tokens.pkl`.

Optional plain-text export (one document per line):

```bash
uv run python -c "import pickle, pathlib; p=pathlib.Path('problem1/output/corpus_tokens.pkl'); d=pickle.loads(p.read_bytes()); pathlib.Path('problem1/output/corpus.txt').write_text('\n'.join(' '.join(s) for s in d)+'\n', encoding='utf-8')"
```

### Task 2 — Word2Vec grid (CBOW + Skip-gram)

Trains the 2×2×2 hyperparameter grid per architecture; writes `problem1/output/w2v_experiments.csv`, manifest JSON, and `problem1/output/problem1_task2_report.md`.

```bash
uv run python -m problem1.train_word2vec
```

Useful flags:

- **`--quick`** — tiny toy corpus + small grid (fast smoke test).
- **`--rebuild-corpus`** — delete corpus pickle before loading.
- **`--save-models`** — save each gensim model under `problem1/output/w2v_checkpoints/` (large).
- **`--epochs N`** — override default epochs (25 full / 8 quick).

### Task 3 — Semantic analysis (neighbors + analogies)

Trains one CBOW and one Skip-gram with fixed hparams (`vector_size=200`, `window=10`, `negative=15` by default). Writes:

- `problem1/output/task3_neighbors.csv`
- `problem1/output/task3_semantic_analysis.md` (ignored by git if you use the assignment `.gitignore`)

```bash
uv run python -m problem1.task3_semantic
```

Flags: **`--quick`**, **`--rebuild-corpus`**, **`--vector-size`**, **`--window`**, **`--negative`**, **`--epochs`**, **`--seed`**.

### Task 4 — PCA / t-SNE plots

Needs Task 3–style models (retrains CBOW + Skip-gram internally). Writes PNGs under `problem1/output/plots/` and `problem1/output/task4_visualization.md`.

```bash
uv run python -m problem1.task4_visualize
```

Flags: **`--quick`**, **`--rebuild-corpus`**, **`--perplexity`** (t-SNE).

---

## Problem 2

Install Problem 2 extras if you have not:

```bash
uv sync --extra problem2
```

### Task 0 — Training names list

**With Gemini:**

```bash
uv run python problem2/generate_training_names.py -n 1000 --batch-size 250 --out problem2/data/TrainingNames.txt --timeout-sec 240 --heartbeat-sec 20
```

**Offline (no API):**

```bash
uv run python problem2/generate_training_names.py --mock -n 1000 --out problem2/data/TrainingNames.txt
```

### Task 1–2 — Train all three models (RNN, Prefix BLSTM, RNN+Attention)

From repo root; uses GPU if available:

```bash
uv run python -m problem2.colab_train_all \
  --data problem2/data/TrainingNames.txt \
  --out problem2/problem2_colab_out \
  --epochs 40 --batch-size 32 --lr 0.002 \
  --embed-dim 64 --hidden-dim 128 \
  --models rnn,blstm,attn
```

Outputs: `checkpoint_*.pt`, `loss_curve_*.png`, `loss_history_*.json`, `train_meta_*.json`, `training_summary.json`.

**Train a single architecture:**

```bash
uv run python -m problem2.train --data problem2/data/TrainingNames.txt --out problem2/problem2_colab_out --model rnn --epochs 40
```

(`--model`: `rnn` | `blstm` | `attn`)

### Task 2 (metrics) — Novelty & diversity + samples

All three checkpoints:

```bash
uv run python -m problem2.run_eval_all \
  --train-data problem2/data/TrainingNames.txt \
  --checkpoint-dir problem2/problem2_colab_out \
  --n 500
```

One checkpoint:

```bash
uv run python -m problem2.evaluate \
  --checkpoint problem2/problem2_colab_out/checkpoint_rnn.pt \
  --train-data problem2/data/TrainingNames.txt \
  --out-json problem2/problem2_colab_out/eval_rnn.json
```

---

## Suggested full order

1. `uv sync` and `uv sync --extra problem2`
2. **P1:** `build_corpus` → `train_word2vec` → `task3_semantic` → `task4_visualize`
3. **P2:** `generate_training_names` → `colab_train_all` → `run_eval_all`

---

## Layout

| Path | Role |
|------|------|
| `problem1/build_corpus.py` | Crawl + tokenized corpus |
| `problem1/train_word2vec.py` | Task 2 grid |
| `problem1/task3_semantic.py` | Task 3 |
| `problem1/task4_visualize.py` | Task 4 |
| `problem1/domain_analogies.txt` | Domain analogy file for evaluation |
| `problem2/models.py` | Char RNN / BLSTM / attention |
| `problem2/colab_train_all.py` | Train all three models |
| `problem2/evaluate.py` | Generation metrics |
| `problem2/generate_training_names.py` | Task 0 names (Gemini or mock) |

---

## Git note

If this repository only tracks `problem1/` and `problem2/` plus `.gitignore`, add this file explicitly when you want it on GitHub:

```bash
git add README.md
git commit -m "Add README runbook"
git push
```
