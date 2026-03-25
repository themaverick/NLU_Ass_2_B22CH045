# Problem 1 — Task 2: Word2Vec experiments

## Data (from `corpus_meta.json`)

| Statistic | Value |
| --- | ---: |
| Documents | 915 |
| Tokens | 192621 |
| Vocabulary (types) | 13846 |

## Design

I train **CBOW** (`sg=0`) and **Skip-gram** (`sg=1`) with **negative sampling**. I vary three categories with **at least two settings each**:

- **Embedding size** (`vector_size`): [128, 256]
- **Context window** (`window`): [5, 10]
- **Negative samples** (`negative`): [10, 20]

That yields a full factorial over those grids (each architecture). I fix `min_count=2`, `epochs=25`, `ns_exponent=0.75`.

## Results (all runs)

| architecture | vector_size | window | negative | training_loss | google_analogy_acc | domain_analogy_acc | domain_pair_sim | domain_pairs_used | train_seconds | vocab_size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CBOW | 128 | 5 | 10 | 1610715.625000 | 0.004386 | 0.200000 | 0.312772 | 7 | 2.2 | 10145 |
| Skip-gram | 128 | 5 | 10 | 7807942.000000 | 0.002193 | 0.000000 | 0.375926 | 7 | 9.03 | 10145 |
| CBOW | 128 | 5 | 20 | 1774318.125000 | 0.002193 | 0.000000 | 0.268966 | 7 | 3.71 | 10145 |
| Skip-gram | 128 | 5 | 20 | 9580040.000000 | 0.002193 | 0.000000 | 0.356536 | 7 | 20.81 | 10145 |
| CBOW | 128 | 10 | 10 | 1617987.250000 | 0.000000 | 0.000000 | 0.340944 | 7 | 3.76 | 10145 |
| Skip-gram | 128 | 10 | 10 | 14416260.000000 | 0.000000 | 0.200000 | 0.378863 | 7 | 24.94 | 10145 |
| CBOW | 128 | 10 | 20 | 1820454.625000 | 0.002193 | 0.000000 | 0.272570 | 7 | 5.8 | 10145 |
| Skip-gram | 128 | 10 | 20 | 18083214.000000 | 0.000000 | 0.200000 | 0.394943 | 7 | 50.68 | 10145 |
| CBOW | 256 | 5 | 10 | 1903811.500000 | 0.002193 | 0.000000 | 0.314777 | 7 | 4.98 | 10145 |
| Skip-gram | 256 | 5 | 10 | 9202089.000000 | 0.002193 | 0.000000 | 0.330236 | 7 | 20.39 | 10145 |
| CBOW | 256 | 5 | 20 | 2122449.250000 | 0.006579 | 0.000000 | 0.273074 | 7 | 8.11 | 10145 |
| Skip-gram | 256 | 5 | 20 | 11276896.000000 | 0.000000 | 0.000000 | 0.325101 | 7 | 37.09 | 10145 |
| CBOW | 256 | 10 | 10 | 1799222.625000 | 0.002193 | 0.200000 | 0.341290 | 7 | 5.25 | 10145 |
| Skip-gram | 256 | 10 | 10 | 16445476.000000 | 0.000000 | 0.000000 | 0.308964 | 7 | 34.54 | 10145 |
| CBOW | 256 | 10 | 20 | 2253263.750000 | 0.000000 | 0.000000 | 0.275945 | 7 | 7.89 | 10145 |
| Skip-gram | 256 | 10 | 20 | 18325516.000000 | 0.000000 | 0.000000 | 0.306518 | 7 | 62.05 | 10145 |

Full table: `w2v_experiments.csv`. Manifest: `w2v_experiment_manifest.json`.

## Best runs (by intrinsic scores)

| Model | Best on Google analogies | Best on domain analogies |
| --- | --- | --- |
| CBOW | `d=256, window=5, negative=20` (google_analogy_acc=0.0066) | `d=128, window=5, negative=10` (domain_analogy_acc=0.2000) |
| Skip-gram | `d=128, window=5, negative=10` (google_analogy_acc=0.0022) | `d=128, window=10, negative=10` (domain_analogy_acc=0.2000) |

## Notes

- I compare `training_loss` only across runs with the **same** `negative` and `epochs` (gensim scales loss with noise count).
- Google analogy accuracy is often **low** on a narrow domain corpus; I still use it **relatively** across runs.
- I run Task 3–4 scripts separately; they read the same cached corpus under `problem1/output/`.
