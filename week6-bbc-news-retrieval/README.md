# BBC News Information Retrieval System
TECH 400 assignment by Shreya Tiwari.

A Python implementation of Boolean retrieval, TF-IDF cosine similarity and BM25 over BBC news articles. It uses shared lowercasing, alphabetic tokenization and English stop-word removal. An inverted index stores term frequencies. Boolean operators have precedence NOT > AND > OR and support parentheses. Ranked models return only positive scores with deterministic ID tie breaking.

## Dataset
Kaggle: https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category

File: `bbc-text.csv`, columns `category` and `text`. Raw: 2,225 articles. Remove 99 exact duplicates to index 2,126 articles. Category labels are used only for the evaluation proxy, never for scoring. The article corpus is not redistributed in this repository. Download it from Kaggle and respect its terms and BBC rights. Dataset SHA-256 is recorded in `results/results.json`.

## Run in Windows PowerShell or a terminal
From this folder:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_data.py
python search.py "mobile phone technology" --model bm25
python search.py "mobile phone technology" --model tfidf
python search.py "mobile AND NOT phone" --model boolean
python experiment.py
python -m unittest -v
```
On macOS/Linux activate with `source .venv/bin/activate` instead. If Kaggle blocks the automatic download, download the ZIP from the dataset page and place `bbc-text.csv` in `data/`. Empty, stop-word-only and out-of-vocabulary ranked queries return no results. Boolean queries require explicit uppercase operators; malformed expressions raise a clear error.

## Files
- `search.py`: loading, preprocessing, inverted index, Boolean parser, TF-IDF and BM25.
- `experiment.py`: dataset statistics, example queries and ten-query evaluation.
- `download_data.py`: reproducible public download with checksum validation.
- `test_search.py`: five meaningful unit tests, including BM25 formula verification.
- `results/results.json`: actual experiment output.
- `results-tests.txt`: actual test output.

## Evaluation limits
Precision@5, nDCG@5 and reciprocal rank use the target news category as a coarse relevance proxy for ten manually selected queries, two per category. These are illustrative results, not topic-level human relevance judgments or a standard benchmark. All documents are indexed because this is unsupervised corpus retrieval; no supervised train/test split is needed. Parameters were not tuned on these queries. Scores from different models are not directly comparable.

BM25 uses k1=1.5, b=0.75 and positive smoothed IDF `log(1+(N-df+0.5)/(df+0.5))`. Query terms are counted once. TF-IDF uses sublinear term frequency, smoothed IDF and L2 normalization. No stemming, synonyms or phrase retrieval are implemented.
