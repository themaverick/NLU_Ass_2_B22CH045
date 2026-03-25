# Problem 1 — Task 3: Semantic analysis

I trained **CBOW** and **Skip-gram** with `vector_size=200`, `window=10`, `negative=15`, `epochs=25`.

## Top-5 cosine neighbors

| Query | Resolved | Model | Rank | Neighbor | Cosine |
| --- | --- | --- | ---: | --- | ---: |
| research | research | CBOW | 1 | dst | 0.5387 |
| research | research | CBOW | 2 | synthase | 0.5035 |
| research | research | CBOW | 3 | organic | 0.4616 |
| research | research | CBOW | 4 | cooling | 0.4608 |
| research | research | CBOW | 5 | chemistry | 0.4553 |
| research | research | Skip-gram | 1 | sponsored | 0.6220 |
| research | research | Skip-gram | 2 | project | 0.5663 |
| research | research | Skip-gram | 3 | older | 0.5317 |
| research | research | Skip-gram | 4 | arun | 0.5238 |
| research | research | Skip-gram | 5 | cooling | 0.5134 |
| student | student | CBOW | 1 | summer | 0.6124 |
| student | student | CBOW | 2 | programme | 0.5803 |
| student | student | CBOW | 3 | semester | 0.5393 |
| student | student | CBOW | 4 | register | 0.5336 |
| student | student | CBOW | 5 | registration | 0.5297 |
| student | student | Skip-gram | 1 | falls | 0.5027 |
| student | student | Skip-gram | 2 | accumulate | 0.5003 |
| student | student | Skip-gram | 3 | register | 0.4976 |
| student | student | Skip-gram | 4 | manjari | 0.4976 |
| student | student | Skip-gram | 5 | petition | 0.4974 |
| phd | phd | CBOW | 1 | dst-inspire | 0.7938 |
| phd | phd | CBOW | 2 | mentor | 0.7290 |
| phd | phd | CBOW | 3 | fellowship | 0.6979 |
| phd | phd | CBOW | 4 | perovskite | 0.6812 |
| phd | phd | CBOW | 5 | physics | 0.6720 |
| phd | phd | Skip-gram | 1 | dst-inspire | 0.6446 |
| phd | phd | Skip-gram | 2 | bhagat | 0.5560 |
| phd | phd | Skip-gram | 3 | perovskite | 0.5531 |
| phd | phd | Skip-gram | 4 | mentor | 0.5493 |
| phd | phd | Skip-gram | 5 | simultaneous | 0.5345 |
| exam | exam | CBOW | 1 | marital | 0.7306 |
| exam | exam | CBOW | 2 | bill | 0.7130 |
| exam | exam | CBOW | 3 | status | 0.7116 |
| exam | exam | CBOW | 4 | belongs | 0.7094 |
| exam | exam | CBOW | 5 | id | 0.7089 |
| exam | exam | Skip-gram | 1 | marital | 0.7911 |
| exam | exam | Skip-gram | 2 | belongs | 0.7805 |
| exam | exam | Skip-gram | 3 | div | 0.7761 |
| exam | exam | Skip-gram | 4 | url | 0.7337 |
| exam | exam | Skip-gram | 5 | passed | 0.7098 |

## Analogies (vector offset)

### `ug : btech :: pg : ?`

**CBOW** (tokens `ug`, `btech`, `pg`)

| Rank | Token | Cosine |
| ---: | --- | ---: |
| 1 | diploma | 0.6092 |
| 2 | undergraduation | 0.6088 |
| 3 | bsc | 0.6027 |
| 4 | pgd | 0.5665 |
| 5 | opting | 0.5359 |

**Skip-gram** (tokens `ug`, `btech`, `pg`)

| Rank | Token | Cosine |
| ---: | --- | ---: |
| 1 | undergraduation | 0.6689 |
| 2 | bsc | 0.6399 |
| 3 | opting | 0.6049 |
| 4 | pgd | 0.5803 |
| 5 | diploma | 0.5783 |

### `undergraduate : bachelor :: graduate : ?`

**CBOW** (tokens `undergraduate`, `bachelor`, `graduate`)

| Rank | Token | Cosine |
| ---: | --- | ---: |
| 1 | four-year | 0.6058 |
| 2 | bsc | 0.5964 |
| 3 | undergraduation | 0.5392 |
| 4 | mca | 0.5372 |
| 5 | diploma | 0.5310 |

**Skip-gram** (tokens `undergraduate`, `bachelor`, `graduate`)

| Rank | Token | Cosine |
| ---: | --- | ---: |
| 1 | fouryeardurationinengineering | 0.5135 |
| 2 | four-year | 0.4958 |
| 3 | nbhm | 0.4800 |
| 4 | waqf | 0.4748 |
| 5 | ora | 0.4684 |

### `faculty : professor :: student : ?`

**CBOW** (tokens `faculty`, `professor`, `student`)

| Rank | Token | Cosine |
| ---: | --- | ---: |
| 1 | drc | 0.4789 |
| 2 | awarded | 0.4681 |
| 3 | ac | 0.4515 |
| 4 | grades | 0.4433 |
| 5 | associate | 0.4376 |

**Skip-gram** (tokens `faculty`, `professor`, `student`)

| Rank | Token | Cosine |
| ---: | --- | ---: |
| 1 | managed | 0.4988 |
| 2 | arranged | 0.4938 |
| 3 | independentstudycourseswiththe | 0.4921 |
| 4 | nominatedbytheheadof | 0.4802 |
| 5 | associate | 0.4730 |

### `course : credit :: degree : ?`

**CBOW** (tokens `course`, `credit`, `degree`)

| Rank | Token | Cosine |
| ---: | --- | ---: |
| 1 | minimum | 0.6926 |
| 2 | requirements | 0.5854 |
| 3 | joining | 0.5611 |
| 4 | convenience | 0.5603 |
| 5 | graduation | 0.5594 |

**Skip-gram** (tokens `course`, `credit`, `degree`)

| Rank | Token | Cosine |
| ---: | --- | ---: |
| 1 | petitioning | 0.4842 |
| 2 | dual | 0.4515 |
| 3 | happen | 0.4433 |
| 4 | accumulation | 0.4282 |
| 5 | bachelor | 0.4233 |

### `science : engineering :: theory : ?`

**CBOW** (tokens `science`, `engineering`, `theory`)

| Rank | Token | Cosine |
| ---: | --- | ---: |
| 1 | discrete | 0.4952 |
| 2 | automata | 0.4882 |
| 3 | textbooks | 0.4836 |
| 4 | tamassia | 0.4827 |
| 5 | bass | 0.4822 |

**Skip-gram** (tokens `science`, `engineering`, `theory`)

| Rank | Token | Cosine |
| ---: | --- | ---: |
| 1 | partially | 0.3551 |
| 2 | motionestimation | 0.3443 |
| 3 | mao | 0.3394 |
| 4 | turbo | 0.3377 |
| 5 | chapman | 0.3372 |

**Takeaway:** On a domain corpus, neighbors track **co-occurrence**; analogies need all tokens in vocabulary.
