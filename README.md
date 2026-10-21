# Sentiment Analysis with a Domain Lexicon

## The problem

General-purpose sentiment models misread healthcare feedback. In everyday
English, words like *wait*, *hours*, *parking*, or *insurance* are neutral, but in a
patient survey comment they nearly always signal frustration. Words like
*thorough* or *listened* are some of the strongest praise a clinic can get. An
off-the-shelf scorer misses this, so it labels many genuinely negative comments
as neutral or positive.

This project scores free-text survey comments, labels each one positive or
negative, and flags statistically unusual comments for follow-up.

## The approach

1. **NLTK VADER as the base.** [VADER](https://github.com/cjhutto/vaderSentiment)
   is a rule-based, lexicon-driven sentiment analyzer. It handles negation,
   intensifiers ("very", "extremely"), capitalization, and punctuation emphasis,
   and it needs no training data.
2. **A custom domain lexicon on top.** `health_lexicon.py` holds about 120
   hand-tuned word weights that are merged into VADER's lexicon at scoring time.
   They override or extend VADER's defaults with healthcare-specific meanings,
   e.g. `wait: -0.7`, `parking: -1.7`, `thorough: 1.7`.
3. **Sentence-level scoring.** Each comment is split into sentences. Each
   sentence gets a VADER compound score, and the scores are summed, so a long
   comment with several complaints is not diluted by one polite sentence.
4. **Normalization.** The summed score is mapped to the range (-1, 1) with
   `score / sqrt(score² + 10)`.
5. **Labeling and anomaly flags.** Scores ≥ 0 are labeled *Positive*, and
   scores < 0 *Negative*. Comments more than 3 standard deviations from the
   mean are flagged *Out of Control* for review.

## How to run it

Requires Python 3.12+.

```bash
git clone https://github.com/adamheisey22/sentiment_analysis_lexicon.git
cd sentiment_analysis_lexicon
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then, from the repository root:

```python
import pandas as pd
import sentiment_analyzer   # first import downloads the NLTK data it needs

df = pd.DataFrame({"comments": [
    "The staff were professional and thorough. Great visit!",
    "Waited two hours. Parking was terrible.",
]})

scored = sentiment_analyzer.get_sentiment_analysis(df, "comments")
print(scored[["comments", "OVERALL_SCORE", "OVERALL_SENTIMENT"]])
```

`get_sentiment_analysis(dataframe, column_name)` takes a DataFrame and the name
of its text column. It returns the same DataFrame with these columns added:

| Column | Description |
|---|---|
| `SENTENCE_COMP` | Dict of `{sentence: compound score}` for each sentence |
| `SENTENCE_NUM` | Number of sentences in the comment |
| `SUM_SCORES` | Sum of the sentence compound scores |
| `OVERALL_SCORE` | `SUM_SCORES` normalized to (-1, 1) |
| `OVERALL_SENTIMENT` | `"Positive"` if `OVERALL_SCORE >= 0`, else `"Negative"` |
| `SENTIMENT_IND` | 1 = Positive, 0 = Negative |
| `TEXT_WORDS` | Comment tokens with stopwords and words of 2 or fewer characters removed |
| `VIOL_DSC` | `"Out of Control"` if the score is beyond ±3σ of the mean, else `"In Control"` |
| `VIOL_IND` | 1 = Out of Control, 0 = In Control |

Missing comments are filled with `"No comment"` before scoring.

### Tuning the lexicon

Add or adjust entries in the `health_lexicon` dict in `health_lexicon.py`.
Weights use VADER's scale (roughly -4 to +4). Keys are case-sensitive.

### Behind a corporate proxy

If `nltk.download` can't reach the internet, configure your proxy with
`nltk.set_proxy(...)` before importing `sentiment_analyzer`. Keep credentials
in environment variables, never in the notebook.

## Repository layout

| File | Purpose |
|---|---|
| `sentiment_analyzer.py` | Scoring pipeline and evaluation helpers (`results_matrix` prints a classification report against labeled data) |
| `health_lexicon.py` | Custom domain lexicon |
| `sentiment_notebook_local.ipynb` | Local notebook walkthrough (loading data is left as a placeholder) |
| `sentiment_notebook.py` | Databricks/PySpark version that reads a table and writes the scored results to a Delta table |
| `dev/text_results.py` | Exploratory plotting helpers (also needs `plotly` and `matplotlib`) |

## Citations

- Bird, Steven, Edward Loper and Ewan Klein (2009). *Natural Language Processing
  with Python.* O'Reilly Media Inc.
- Hutto, C.J. and Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for
  Sentiment Analysis of Social Media Text. *Eighth International Conference on
  Weblogs and Social Media (ICWSM-14).*

## Contact

Questions: **Adam Heisey**, `admrussel@gmail.com`
