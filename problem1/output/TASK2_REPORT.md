# Task 2: Word2Vec model training — report

## 1. Objective

Train two **Word2Vec** architectures with **negative sampling** on the Task 1 corpus:

- **CBOW** (`sg=0`) predicts a target word from averaged context embeddings.
- **Skip-gram** (`sg=1`) predicts context words from the target embedding.

For each architecture, systematically vary **embedding dimension** (`vector_size`), **context window** (`window`), and **number of negative samples** (`negative`), record training loss and intrinsic evaluation scores, and compare configurations.

## 2. Data

| Statistic | Value |
| --- | ---: |
| Documents (tokenized units from Task 1) | 9 |
| Total tokens | 29 |
| Vocabulary (types, Task 1 tokenizer) | 12 |

Training uses `min_count=2`, `epochs=8`, `sorted_vocab=True`, and the same lowercased token strings as Task 1.

## 3. Methodology

- **Implementation:** `gensim.models.Word2Vec` (Mikolov-style skip-gram / CBOW with negative sampling). Reported `training_loss` is gensim’s running negative-sampling loss; it scales with `negative`, so compare loss **only** across runs with the **same** `negative` (and same `epochs`).
- **Negative sampling:** `negative` controls how many “noise” draws per positive pair; `ns_exponent=0.75` matches the smoothed unigram distribution used in the original work.
- **Embedding dimension:** larger `vector_size` increases representational capacity but, on a **small** corpus, can overfit or fragment the frequency budget across dimensions.
- **Context window:** larger `window` mixes more distant co-occurrences (broader “semantic” context) but dilutes the immediate predictive signal; Skip-gram often benefits more from wide windows than CBOW because each (target, context) pair is a separate training example.
- **Evaluation (intrinsic):**
  - **Google analogy benchmark** (`questions-words.txt` shipped with gensim): many items are **out-of-vocabulary** on a narrow domain corpus, so **absolute accuracy is often near zero**; it still offers a **comparable relative** signal across runs with identical evaluation settings (`restrict_vocab=40000`, case-insensitive).
  - **Domain analogies** (`domain_analogies.txt`): morphology and academic-style relations using vocabulary more likely to appear in institute text.
  - **Domain pair similarity:** mean cosine similarity between hand-picked related pairs (only pairs where both words exist in the model vocabulary are averaged; `domain_pairs_used` records how many contributed).

Full numeric results are in `task2_results.csv`.

## 4. Results (all runs)

| architecture | vector_size | window | negative | training_loss | google_analogy_acc | domain_analogy_acc | domain_pair_sim | domain_pairs_used | train_seconds | vocab_size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CBOW | 64 | 3 | 5 | 125.614000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0.01 | 12 |
| Skip-gram | 64 | 3 | 5 | 32.494900 | 0.000000 | 0.000000 | 0.000000 | 0 | 0.01 | 12 |
| CBOW | 64 | 3 | 15 | 338.651500 | 0.000000 | 0.000000 | 0.000000 | 0 | 0.01 | 12 |
| Skip-gram | 64 | 3 | 15 | 84.815100 | 0.000000 | 0.000000 | 0.000000 | 0 | 0.01 | 12 |
| CBOW | 64 | 6 | 5 | 82.534900 | 0.000000 | 0.000000 | 0.000000 | 0 | 0.01 | 12 |
| Skip-gram | 64 | 6 | 5 | 14.815700 | 0.000000 | 0.000000 | 0.000000 | 0 | 0.01 | 12 |
| CBOW | 64 | 6 | 15 | 222.897200 | 0.000000 | 0.000000 | 0.000000 | 0 | 0.01 | 12 |
| Skip-gram | 64 | 6 | 15 | 42.374100 | 0.000000 | 0.000000 | 0.000000 | 0 | 0.01 | 12 |
| CBOW | 128 | 3 | 5 | 125.614000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0.01 | 12 |
| Skip-gram | 128 | 3 | 5 | 32.507100 | 0.000000 | 0.000000 | 0.000000 | 0 | 0.01 | 12 |
| CBOW | 128 | 3 | 15 | 338.608900 | 0.000000 | 0.000000 | 0.000000 | 0 | 0.01 | 12 |
| Skip-gram | 128 | 3 | 15 | 84.766400 | 0.000000 | 0.000000 | 0.000000 | 0 | 0.01 | 12 |
| CBOW | 128 | 6 | 5 | 82.522700 | 0.000000 | 0.000000 | 0.000000 | 0 | 0.01 | 12 |
| Skip-gram | 128 | 6 | 5 | 14.821800 | 0.000000 | 0.000000 | 0.000000 | 0 | 0.01 | 12 |
| CBOW | 128 | 6 | 15 | 222.854500 | 0.000000 | 0.000000 | 0.000000 | 0 | 0.01 | 12 |
| Skip-gram | 128 | 6 | 15 | 42.325300 | 0.000000 | 0.000000 | 0.000000 | 0 | 0.01 | 12 |

## 5. Analysis

### 5.1 Best configurations (by metric)

| Model | Best on Google analogies | Best on domain analogies |
| --- | --- | --- |
| CBOW | `d=64, window=3, negative=5` (google_analogy_acc=0.0000) | `d=64, window=3, negative=5` (domain_analogy_acc=0.0000) |
| Skip-gram | `d=64, window=3, negative=5` (google_analogy_acc=0.0000) | `d=64, window=3, negative=5` (domain_analogy_acc=0.0000) |

### 5.2 Embedding dimension (`vector_size`)

Increasing dimensionality raises model capacity. On **small** corpora, very large embeddings can memorize idiosyncratic co-occurrences without improving general analogy or similarity structure; mid-range dimensions (e.g. 100–300) are a common compromise. In your table, compare rows that share the same `window` and `negative` to isolate the effect of `vector_size` on `training_loss` and analogy scores.

### 5.3 Context window (`window`)

A **narrow** window emphasizes syntactic and immediate collocations (e.g. multi-word phrases). A **wide** window pulls in more document-level co-occurrence signal. Skip-gram typically scales better with larger windows because it emits more independent (center, context) training pairs per sentence. CBOW averages context vectors, so an overly large window can **blur** the context representation, sometimes hurting fine-grained prediction.

### 5.4 Negative samples (`negative`)

More negative samples approximate the softmax denominator more sharply and can stabilize training, but each additional negative increases work per positive example. Values around **5–25** are standard; if `training_loss` and analogy metrics plateau, increasing `negative` further may yield diminishing returns.

### 5.5 CBOW vs Skip-gram

**Skip-gram** tends to perform better on **rare words** because it generates more training updates per rare token. **CBOW** is often faster and can be stronger for **frequent** words when data are abundant. On small domain corpora, Skip-gram is frequently the stronger default for semantic retrieval and analogy-style tests, but the winning configuration should be read from your table rather than assumed.

## 6. Conclusion

This experiment grid isolates the effects of **vector size**, **context window**, and **negative sampling** under a fixed tokenizer and corpus. Use the **relative** ordering of `domain_analogy_acc`, `domain_pair_sim`, and `training_loss` (together with qualitative checks such as `model.wv.most_similar`) to choose a deployment configuration for downstream NLU tasks. For publication-style claims, complement intrinsic scores with an **extrinsic** task (e.g. classification or clustering) on the same domain.

---
*Generated by `task2.py`. Re-run after changing `SOURCE_URLS`, using `--rebuild-corpus`, or editing the hyperparameter grids in `task2.py`. If `task3_4.py` has been run, its appendix between HTML comment markers is preserved across Task 2 regenerations.*
