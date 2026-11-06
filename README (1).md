# Week 4: Compare, Cross-Validate and Tune Models

Classification of Urdu news items as **FAKE** or **TRUE** using KNN, Decision Tree, Random Forest and Gradient Boosting on TF-IDF text features, with 5-fold cross-validation, hyper-parameter tuning and an overfitting check.

## Files

| File | Description |
|---|---|
| `Week4_Model_Comparison.ipynb` | Main Google Colab notebook (all 6 steps + Gradient Boosting bonus) |
| `urdu_news.csv` | Dataset (Urdu news headlines/articles with FAKE / TRUE labels) |
| `Week4_Report.docx` | Written report with tables, plots and answers |
| `README.md` | This file |

## Dataset

- **Source:** `urdu_news.csv` (Urdu fake news dataset)
- **Raw rows / columns:** 10,106 rows / 3 columns (`Sr. No.`, `News Items`, `Label`)
- **Text column:** `News Items` (Urdu text, from short headlines of about 3 words to long articles of about 1,200 words; median about 30 words)
- **Target:** `Label` (2 classes, balanced after cleaning: FAKE 4,637, TRUE 4,598)
- **Dropped column:** `Sr. No.` (row index, no predictive meaning)

### Data cleaning applied

The raw file has several problems that the notebook fixes before modelling:

| Issue found | Fix |
|---|---|
| 22 completely empty rows at the end of the file | Dropped |
| 1 repeated header row inside the data (`Sr. No., News Items, Label`) | Dropped |
| Inconsistent labels: `FAKE `, `TRUE ` (trailing spaces) | Stripped and upper-cased to `FAKE` / `TRUE` |
| `Sr. No.` restarts partway through the file (only 5,053 unique values for 10,083 rows) | Column dropped, not used as a feature |
| 848 duplicate news texts (no text has conflicting labels) | Duplicates removed so the same article cannot appear in both train and test |
| Extra spaces, stray Latin characters (e.g. ` CPEC `) | Whitespace normalised, text kept as is |

After cleaning: **9,235 unique rows**, 2 balanced classes.

## Features

- Text is converted to numbers with `TfidfVectorizer` (word 1-2 grams, `max_features` limited, no English stop-word list because the text is Urdu).
- The vectorizer is placed **inside each model Pipeline**, so it is fitted only on the training part of each CV fold (prevents data leakage).
- KNN uses cosine distance on the TF-IDF vectors; tree models use the TF-IDF vectors directly.

## How to run (Google Colab)

1. Open [colab.research.google.com](https://colab.research.google.com) and upload `Week4_Model_Comparison.ipynb` (File → Upload notebook).
2. Run the first code cell and upload `urdu_news.csv` when asked (or put the CSV in the Colab file panel first).
3. Runtime → **Run all**. Total run time is a few minutes to ten minutes (TF-IDF on 9k texts makes the Random Forest and Gradient Boosting grids the slowest part).

Required libraries (already in Colab): `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

## What the notebook does

| Step | Content |
|---|---|
| 1 | Load and clean data (see table above), encode `FAKE`=0 / `TRUE`=1, 80/20 split (`stratify=y`, `random_state=42`) |
| 2 | 5-fold CV of KNN (TF-IDF + cosine), Decision Tree, Random Forest, Gradient Boosting on training data only |
| 3 | KNN: K = 1, 3, ..., 15. Decision Tree: max_depth = 2, 3, 5, 10, None. Plots + explanation |
| 4 | Random Forest `GridSearchCV` (`n_estimators` x `max_depth`, cv=5) = 9 x 5 = 45 trainings |
| Bonus | Gradient Boosting `GridSearchCV` (`learning_rate` x `n_estimators`) = 12 x 5 = 60 trainings |
| 5 | Train / test / gap table for all tuned models |
| 6 | Final model evaluated once on the test set: accuracy, confusion matrix, classification report (precision / recall / F1 for FAKE and TRUE), decision |

## Results summary

Fill this table in after running the notebook (the numbers from the old healthcare dataset no longer apply).

| Step | Result |
|---|---|
| Best CV model (default settings) | _fill in_ |
| Best K (KNN) | _fill in_ |
| Best max_depth (Decision Tree) | _fill in_ |
| Random Forest best params | _fill in_ |
| Gradient Boosting best params | _fill in_ |
| Most overfit model | _fill in_ (expect KNN with K=1 or an unlimited-depth Decision Tree) |
| Most stable model | _fill in_ |
| Final model | _fill in_ |

## Important notes

- **Baseline:** the classes are balanced, so a random guess scores about 0.50. Any model should be judged against that, not against 0.33 as in the previous dataset.
- **Length bias:** FAKE items are much longer on average (about 116 words) than TRUE items (about 34 words). Models may partly learn text length rather than real fake-news cues. Consider this when interpreting high scores, and optionally compare against a model using only word count.
- **Duplicates:** removing the 848 duplicate texts matters; without it, test accuracy would be inflated because copies of the same article appear in both train and test.
- **Language:** Urdu is written right to left. Display issues in plots or tables are cosmetic only and do not affect the models.

## Reproducibility

All random processes use `random_state=42`, so the notebook gives the same numbers on every run.
