# Project Notes

## Dataset
- Twitter US Airline Sentiment (Kaggle, Crowdflower), 14,640 tweets
- After cleaning (duplicates + empty text removed): 14,452 rows
- Labels: negative 9087 (~63%), neutral 3067 (~21%), positive 2298 (~16%)
- The data is imbalanced, so accuracy alone is misleading. A model that always says "negative" gets ~63% accuracy (macro-F1 ~0.26). I use macro-F1 instead.

## Setup
- 80/20 train/test split, `random_state=42`, `stratify` on the label
- TF-IDF (`ngram_range=(1, 2)`, `min_df=2`) + classifier in one sklearn Pipeline
- `class_weight="balanced"` so the small classes are not ignored

## Results (test set, 2891 tweets)

### LogisticRegression
| class | precision | recall | f1 |
|---|---|---|---|
| negative | 0.88 | 0.84 | 0.86 |
| neutral | 0.58 | 0.63 | 0.61 |
| positive | 0.70 | 0.73 | 0.72 |

accuracy 0.78, macro-F1 0.73

### LinearSVC
| class | precision | recall | f1 |
|---|---|---|---|
| negative | 0.85 | 0.89 | 0.87 |
| neutral | 0.61 | 0.57 | 0.59 |
| positive | 0.74 | 0.68 | 0.71 |

accuracy 0.79, macro-F1 0.72

## What I noticed
- LinearSVC has higher accuracy (0.79 vs 0.78) but lower macro-F1 (0.72 vs 0.73). It leans toward the majority class (negative recall 0.89, positive recall 0.68), and accuracy rewards that.
- Neutral is the weakest class (F1 about 0.6). It has fewer examples and its wording overlaps with negative and positive tweets.
- I saved LogisticRegression because its macro-F1 is slightly higher and it supports `predict_proba`, which the API needs for the confidence threshold.

## Experiments
- `ngram_range` (1, 2) -> (1, 1): macro-F1 dropped from 0.73 to 0.71. Bigrams such as "not good" carry meaning that single words miss, so they help.
- `class_weight="balanced"` removed: recall dropped most for the positive class (it was 0.73 with balanced weights). Without the weights the model favours the majority class (negative) and the small classes get ignored.
- Both experiments were reverted, and `train.py` is back to `ngram_range=(1, 2)` with `class_weight="balanced"`.

## Why 80/20 split?
I train on 80% of the data (about 11,560 tweets) so the model has enough examples to learn from, and keep 20% (2,891 tweets) that it never sees during training. I use `stratify` so both parts have the same label mix (~63% negative).

If I trained on all the data and tested on the same data, the model would already have seen those tweets, so the score would only show how well it memorised them, not how it handles new tweets. The held-out 20% gives an honest estimate of real-world performance.

## Error analysis (evaluate.py)

Baseline (always "negative"): accuracy 0.629, macro-F1 0.257. The model gets accuracy 0.781 and macro-F1 0.728, so it learned real signal.

Confusion matrix (rows = true, columns = predicted):

| | pred_negative | pred_neutral | pred_positive |
|---|---|---|---|
| true_negative | 1533 | 215 | 70 |
| true_neutral | 154 | 388 | 71 |
| true_positive | 60 | 64 | 336 |

What I see:
- Most errors involve neutral. 215 negative tweets were predicted neutral, and 154 neutral tweets were predicted negative. Negative vs positive confusion is small (70 and 60).
- The 215 negative-to-neutral errors are the main reason neutral precision is low (0.58): negative tweets leak into the neutral prediction.

I read 15 random misclassified tweets. Patterns in this small sample:
- Many negative tweets have no emotional words. They are requests or questions inside a bad situation, e.g. "called ... it says high call volumes ... is there any way someone can contact me". The model calls these neutral, sometimes with confidence around 0.77.
- Very short, tone-dependent tweets ("well here we go", "sweet follow back and i'll get the dm off") get low confidence (about 0.45) and are often wrong.
- Some neutral tweets contain words common in negative tweets ("nothing", "left my phone"), so the model leans negative.
- Some labels look doubtful. "...helped make awesome" is labelled negative, but reads positive.

Takeaway: errors with confidence around 0.45 can be filtered by an API confidence threshold, but a few wrong predictions reach 0.77, so a threshold will not catch everything.

Note: this is based on 15 tweets out of 634 errors, so these are patterns in a sample, not proof.
