# Problem 1 — Task 4: Embedding maps

I plotted **107** types that appear in **both** vocabularies after pooling `word_lists.VIZ_WORDS` and frequency fallbacks.

## CBOW

**PCA (CBOW)** — `task4_pca_cbow.png`. First two components explain ~**20.8%** variance (L2-normalized rows). Colors mark coarse groups (research, people, credentials, assessment, org, tech, other).

**t-SNE (CBOW)** — `task4_tsne_cbow.png`. I set `perplexity=21`. Distances across far clusters are only qualitative.

## Skip-gram

**PCA (Skip-gram)** — `task4_pca_skipgram.png`. First two components explain ~**10.1%** variance (L2-normalized rows). Colors mark coarse groups (research, people, credentials, assessment, org, tech, other).

**t-SNE (Skip-gram)** — `task4_tsne_skipgram.png`. I set `perplexity=21`. Distances across far clusters are only qualitative.

## How I colored points

| id | Theme |
| ---: | --- |
| 0 | research / discovery |
| 1 | people roles |
| 2 | credentials |
| 3 | assessment |
| 4 | organization / teaching |
| 5 | technology |
| 6 | other |

**Reading tip:** PCA shows global linear structure; t-SNE emphasizes local neighborhoods.
