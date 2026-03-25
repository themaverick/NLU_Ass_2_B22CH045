# Problem 2: Character-level Indian name generation (RNN variants)

Training artifacts live under **`problem2/problem2_colab_out/`** (checkpoints, **`loss_curve_*.png`**, **`loss_history_*.json`**, **`train_meta_*.json`**, **`training_summary.json`**). After evaluation: **`eval_*.json`** and **`eval_*.samples.txt`** (see TASK-2–3 below).

## TASK-0: Dataset

- **Content**: **Indian first names only** (one word per line). Gemini is called in batches of **250** names per request; a **set** enforces uniqueness until `n` names are collected.
- **Regenerate with Gemini** via the **`google-genai`** SDK (set `GOOGLE_API_KEY` or `GEMINI_API_KEY`; do not commit secrets):

  ```bash
  uv sync --extra problem2
  uv run python problem2/generate_training_names.py -n 1000 --batch-size 250 --out TrainingNames.txt
  ```

- **Offline / no API**: `uv run python problem2/generate_training_names.py --mock -n 1000 --out TrainingNames.txt`

## TASK-1: Architectures and hyperparameters

### 1) Vanilla RNN (character LM)

- **Architecture**: Character embedding → stacked `tanh` RNN (`batch_first`) → linear layer to vocabulary logits. Training uses teacher forcing: predict `x[t+1]` from hidden state after `x[:t+1]`. Generation samples from the softmax of the last-step logits autoregressively until the end token or max length.
- **Trainable parameters**: **30,236** (from `train_meta_rnn.json`) — order `V·E + E·H + H² + H·V` for one layer (plus biases), with vocabulary size `V=28`, embedding size `E=64`, hidden size `H=128`.

### 2) Prefix BLSTM (causal bidirectional context)

- **Architecture**: Two separate LSTMs: forward LSTM on the **prefix** `x[0:t+1]` and backward LSTM on the **reversed prefix**. Their last-layer hidden states are concatenated (`2H`) and passed through a linear classifier to predict the next character. This avoids peeking at future characters while still using “backward over the prefix” context. Generation recomputes both LSTMs on the growing prefix each step.
- **Trainable parameters**: **207,644** (from `train_meta_blstm.json`) — two LSTMs plus output layer; order `O(V·E + 2(E·4H + …) + 2H·V)` (LSTM has four gates per direction).

### 3) RNN + additive attention

- **Architecture**: Single-layer `tanh` RNN produces hidden states `h_0…h_t` for the prefix. At step `t`, **query** = current `h_t`, **keys/values** = all prefix states; energies are `vᵀ tanh(W [h_i ; h_t])` (Bahdanau-style), softmax → attention weights, context = weighted sum of prefix states. The classifier input is `[h_t ; context]` (dimension `2H`). Generation mirrors training: cache per-step RNN outputs and recompute attention over the cache.
- **Trainable parameters**: **66,844** (from `train_meta_attn.json`).

### Shared hyperparameters (default in notebook / CLI)

| Setting        | Value (this run, see `training_summary.json`) |
|----------------|--------------------------------|
| Embedding dim  | 64                             |
| Hidden size    | 128                            |
| RNN/BLSTM layers | 1 (per direction for BLSTM)  |
| Learning rate  | 0.02                           |
| Optimizer      | Adam                           |
| Batch size     | 32                             |
| Epochs         | 40                             |
| Grad clip      | 5.0                            |

*(Default Colab/CLI learning rate is **0.002**; this Colab run used **0.02**.)*

## TASK-2: Quantitative comparison

| Model   | Trainable params | Novelty rate | Diversity |
|---------|------------------|--------------|-----------|
| RNN     | 30,236           | 0.928        | 0.916     |
| BLSTM   | 207,644          | 0.052        | 0.730     |
| RNN+Attn| 66,844           | 0.896        | 0.882     |

*Metrics from `python -m problem2.run_eval_all` with `--n 500`, `--temperature 0.85`, training file `problem2/data/TrainingNames.txt` (same vocabulary as training).*

**Definitions**

- **Novelty rate**: fraction of generated names whose normalized string does not appear in the training set.
- **Diversity**: `(unique generated names) / (total generated names)`.

## TASK-3: Qualitative analysis

### Realism

- **RNN**: Mostly short Latin-script tokens that *sound* name-like (`kalbir`, `meenal`, `suresh`) mixed with odd spellings (`chevashinal`, `rivindeep`). Length varies; many are plausible as stylized Indian given names.
- **BLSTM**: Strong overlap with **real, recognizable names** from the training distribution (`rajeshwari`, `kumar`, `amrita`, `anand`). That aligns with the **low novelty** score: the model often emits names that are in or very near the training set.
- **RNN+Attn**: Many names share **prefix patterns** (`rabin`, `ranya`, `rasini`, `raki`…), suggesting attention helps capture substructure but can over-use shared prefixes; still a mix of plausible and garbled forms (`rabivavi`).

### Failure modes

- **RNN / Attn**: Occasional **over-long or fused tokens**, **unlikely letter sequences**, and **repeated morpheme patterns** (especially `ra-` chains in attention samples).
- **BLSTM**: **Memorization / copying** — many generated lines are exact training names or very common Western/Indian names, which **hurts novelty** while improving perceived “realism” for those lines.

### Representative samples

First 12 lines from **`problem2/problem2_colab_out/eval_*.samples.txt`** (500-sample run; first 200 lines saved per file).

**RNN**

```
kakom
jazwan
uzani
chevashinal
sheel
rivindeep
kalbir
soyra
meenal
sughav
shivaan
suresh
```

**Prefix BLSTM**

```
vidushi
om
rajeshwari
raj
gagan
kumar
asif
amrita
leena
anand
naira
ronit
```

**RNN + attention**

```
rabin
ranya
rasini
raki
kania
rimada
rabivavi
reem
noeya
ramin
sana
anav
```

## Files (deliverables)

| Path | Role |
|------|------|
| `TrainingNames.txt` | 1000 training names |
| `problem2/models.py` | Model definitions |
| `problem2/train.py` | Local training CLI |
| `problem2/evaluate.py` | Novelty & diversity + sample dump |
| `problem2/run_eval_all.py` | Runs `evaluate` for all three checkpoints in `problem2_colab_out/` |
| `problem2/generate_training_names.py` | TASK-0 Gemini / mock generator |
| `problem2/problem2_colab_out/` | Colab training outputs + `eval_*.json` / `eval_*.samples.txt` |
| `colab/Problem2_Training.ipynb` | Colab training + metrics |
| `colab/build_problem2_notebook.py` | Rebuilds the `.ipynb` from `problem2/*.py` |

## Local commands (after `uv sync --extra problem2`)

```bash
# Train one model
uv run python -m problem2.train --data TrainingNames.txt --out output/problem2 --model rnn --epochs 40

# Evaluate all three checkpoints under problem2/problem2_colab_out/
uv run python -m problem2.run_eval_all --train-data problem2/data/TrainingNames.txt --checkpoint-dir problem2/problem2_colab_out

# Single checkpoint
uv run python -m problem2.evaluate --checkpoint problem2/problem2_colab_out/checkpoint_rnn.pt --train-data problem2/data/TrainingNames.txt --out-json problem2/problem2_colab_out/eval_rnn.json
```
