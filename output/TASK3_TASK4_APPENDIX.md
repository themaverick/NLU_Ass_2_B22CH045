## 7. Task 3: Semantic analysis (cosine similarity)

### 7.1 Setup

For qualitative analysis we trained **one CBOW** and **one Skip-gram** model with the **same** hyperparameters: `vector_size=100`, `window=6`, `negative=5`, `epochs=25`, `min_count=2`, negative sampling with `ns_exponent=0.75`. **Nearest neighbors** use **cosine** similarity via `gensim.models.KeyedVectors.most_similar`.

### 7.2 Top-5 nearest neighbors

If a query is out-of-vocabulary, we try aliases from `QUERY_ALIASES` (e.g. **exam** → **examination**).

| Query | Resolved | Model | Rank | Neighbor | Cosine |
| --- | --- | --- | ---: | --- | ---: |
| research | research | CBOW | 1 | rfcs | 0.8915 |
| research | research | CBOW | 2 | relevant | 0.8531 |
| research | research | CBOW | 3 | interested | 0.8523 |
| research | research | CBOW | 4 | papers | 0.8475 |
| research | research | CBOW | 5 | drafts | 0.8469 |
| research | research | Skip-gram | 1 | proposal | 0.6258 |
| research | research | Skip-gram | 2 | collaborations | 0.6211 |
| research | research | Skip-gram | 3 | interdisciplinary | 0.6194 |
| research | research | Skip-gram | 4 | years | 0.6156 |
| research | research | Skip-gram | 5 | drafts | 0.6078 |
| student | student | CBOW | 1 | who | 0.9729 |
| student | student | CBOW | 2 | allowed | 0.9683 |
| student | student | CBOW | 3 | candidacy | 0.9618 |
| student | student | CBOW | 4 | completed | 0.9600 |
| student | student | CBOW | 5 | if | 0.9570 |
| student | student | Skip-gram | 1 | casual | 0.7723 |
| student | student | Skip-gram | 2 | repeat | 0.7711 |
| student | student | Skip-gram | 3 | requesting | 0.7686 |
| student | student | Skip-gram | 4 | allowed | 0.7663 |
| student | student | Skip-gram | 5 | attendance | 0.7652 |
| phd | phd | CBOW | 1 | mtech | 0.9888 |
| phd | phd | CBOW | 2 | offered | 0.9759 |
| phd | phd | CBOW | 3 | shortlisted | 0.9549 |
| phd | phd | CBOW | 4 | btech | 0.9504 |
| phd | phd | CBOW | 5 | semester-ii | 0.9317 |
| phd | phd | Skip-gram | 1 | mtech | 0.9055 |
| phd | phd | Skip-gram | 2 | shortlisted | 0.8714 |
| phd | phd | Skip-gram | 3 | semester-ii | 0.8473 |
| phd | phd | Skip-gram | 4 | pre-requisites | 0.8416 |
| phd | phd | Skip-gram | 5 | tech | 0.8358 |
| exam | exam → examination | CBOW | 1 | one | 0.9835 |
| exam | exam → examination | CBOW | 2 | after | 0.9822 |
| exam | exam → examination | CBOW | 3 | comprehensive | 0.9803 |
| exam | exam → examination | CBOW | 4 | additional | 0.9770 |
| exam | exam → examination | CBOW | 5 | under | 0.9763 |
| exam | exam → examination | Skip-gram | 1 | make-up | 0.8678 |
| exam | exam → examination | Skip-gram | 2 | seminar | 0.8667 |
| exam | exam → examination | Skip-gram | 3 | comprehensive | 0.8613 |
| exam | exam → examination | Skip-gram | 4 | candidacy | 0.8583 |
| exam | exam → examination | Skip-gram | 5 | minutes | 0.8396 |

Full long-form table: `task3_neighbors.csv`.

### 7.3 Analogy experiments (vector offset)

For each prompt **A : B :: C : ?** we rank tokens by cosine similarity to **B − A + C**. Abbreviations (**ug**, **pg**, **btech**) map to the first matching synonym present in the vocabulary (see `_alts` in `task3_4.py`).

#### `ug : btech :: pg : ?`

- **CBOW** — vectors for (ug, btech, pg).

| Rank | Prediction | Cosine |
| ---: | --- | ---: |
| 1 | sc | 0.8994 |
| 2 | assignment | 0.8935 |
| 3 | mtech | 0.8904 |
| 4 | tech | 0.8884 |
| 5 | phd | 0.8699 |

- **Skip-gram** — vectors for (ug, btech, pg).

| Rank | Prediction | Cosine |
| ---: | --- | ---: |
| 1 | ee | 0.8705 |
| 2 | l-t-p-c | 0.8673 |
| 3 | mtech | 0.8552 |
| 4 | branches | 0.8448 |
| 5 | bridge | 0.8151 |

#### `undergraduate : bachelor :: graduate : ?`

- **CBOW** — vectors for (undergraduate, bachelor, pg).

| Rank | Prediction | Cosine |
| ---: | --- | ---: |
| 1 | iits | 0.9136 |
| 2 | basagni | 0.9135 |
| 3 | nagaraj | 0.9128 |
| 4 | conti | 0.9084 |
| 5 | maura | 0.9053 |

- **Skip-gram** — vectors for (undergraduate, bachelor, pg).

| Rank | Prediction | Cosine |
| ---: | --- | ---: |
| 1 | inanyrelevantdiscipline | 0.8351 |
| 2 | fouryeardurationinengineering | 0.8066 |
| 3 | applicant | 0.7819 |
| 4 | ora | 0.7811 |
| 5 | happen | 0.7564 |

#### `faculty : professor :: student : ?`

- **CBOW** — vectors for (faculty, professor, student).

| Rank | Prediction | Cosine |
| ---: | --- | ---: |
| 1 | sgpa | 0.9364 |
| 2 | dues | 0.9195 |
| 3 | clearance | 0.9185 |
| 4 | major | 0.9156 |
| 5 | unique | 0.9137 |

- **Skip-gram** — vectors for (faculty, professor, student).

| Rank | Prediction | Cosine |
| ---: | --- | ---: |
| 1 | pg | 0.6644 |
| 2 | normally | 0.6253 |
| 3 | approved | 0.6252 |
| 4 | only | 0.6242 |
| 5 | senate | 0.6209 |

#### `course : credit :: degree : ?`

- **CBOW** — vectors for (course, credit, degree).

| Rank | Prediction | Cosine |
| ---: | --- | ---: |
| 1 | dual | 0.8854 |
| 2 | minimum | 0.8292 |
| 3 | convenience | 0.8230 |
| 4 | primal | 0.8215 |
| 5 | program | 0.8213 |

- **Skip-gram** — vectors for (course, credit, degree).

| Rank | Prediction | Cosine |
| ---: | --- | ---: |
| 1 | dual | 0.7415 |
| 2 | sum | 0.6668 |
| 3 | relaxation | 0.6526 |
| 4 | qualifying | 0.5993 |
| 5 | rules | 0.5989 |

#### `science : engineering :: theory : ?`

- **CBOW** — vectors for (science, engineering, theory).

| Rank | Prediction | Cosine |
| ---: | --- | ---: |
| 1 | graph | 0.8648 |
| 2 | discrete | 0.8579 |
| 3 | visual | 0.8405 |
| 4 | siam | 0.8205 |
| 5 | processing | 0.8193 |

- **Skip-gram** — vectors for (science, engineering, theory).

| Rank | Prediction | Cosine |
| ---: | --- | ---: |
| 1 | blocks | 0.4753 |
| 2 | discrete | 0.4732 |
| 3 | classes | 0.4728 |
| 4 | co-design | 0.4703 |
| 5 | north-holland | 0.4659 |


### 7.4 Discussion (semantic plausibility)

- **Neighbors:** On a **small, domain-specific** corpus, neighbors often reflect **document co-occurrence** (committee names, local collocations) as much as abstract synonymy. CBOW **smooths** context, which can yield **higher cosine** with broad topical associates; Skip-gram **emphasizes** predictive links and often surfaces rarer but informative contexts.

- **Analogies:** Offset analogies assume linear relations between embeddings. They succeed when **A:B** and **C:?** correspond to a **consistent** relation learned from data (e.g. parallel degree naming). Spelling variants (**ug** vs **undergraduate**) and OOV tokens break analogies; when all tokens are in-vocabulary, inspect whether top predictions are **paraphrases**, **siblings in a taxonomy**, or **artifacts** of shared boilerplate.


## 8. Task 4: Visualization (PCA and t-SNE)

### 8.1 Word set

We projected **L2-normalized** embeddings for a fixed list of domain-relevant types that appear in **both** vocabularies (see script `task3_4.py`). Colors mark coarse groups (research, people, credentials, assessment, organization, other).

### 8.2 Figures and captions

![PCA — CBOW](task4_pca_cbow.png)

**Figure (PCA, CBOW).** File `task4_pca_cbow.png`. Two principal components capture about **54.5%** of variance in the selected embedding matrix (L2-normalized rows). Points are colored by coarse category (research/teaching, people, credentials, assessment, org). PCA is linear and preserves global structure; nearby points share major variance directions.


![PCA — Skip-gram](task4_pca_skipgram.png)

**Figure (PCA, Skip-gram).** File `task4_pca_skipgram.png`. Two principal components capture about **24.3%** of variance in the selected embedding matrix (L2-normalized rows). Points are colored by coarse category (research/teaching, people, credentials, assessment, org). PCA is linear and preserves global structure; nearby points share major variance directions.


![t-SNE — CBOW](task4_tsne_cbow.png)

**Figure (t-SNE, CBOW).** File `task4_tsne_cbow.png`. Nonlinear projection (perplexity≈8) of the same vectors. Local neighborhoods are emphasized; distances across disjoint clusters are not strictly comparable.


![t-SNE — Skip-gram](task4_tsne_skipgram.png)

**Figure (t-SNE, Skip-gram).** File `task4_tsne_skipgram.png`. Nonlinear projection (perplexity≈8) of the same vectors. Local neighborhoods are emphasized; distances across disjoint clusters are not strictly comparable.


### 8.3 CBOW vs Skip-gram clustering

- **CBOW** vectors are trained to predict the center from **averaged** context; clusters in PCA often align with **broad topical blobs** (frequent words dominate the average).

- **Skip-gram** updates each context direction separately, which tends to preserve **finer** relational structure; t-SNE may show **tighter** micro-clusters of synonyms or role-related words, but can also separate **low-frequency** types if context evidence is sparse.

- **PCA vs t-SNE:** PCA highlights **global** linear separations; t-SNE highlights **local** neighborhoods. Use PCA for a coarse layout check and t-SNE for neighborhood structure, without over-interpreting **between-cluster** t-SNE distances.
