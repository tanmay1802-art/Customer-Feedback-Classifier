# Project Notes

## Dataset
- Twitter US Airline Sentiment (Kaggle, Crowdflower), 14,640 tweets
- After cleaning (duplicates + empty text removed): 14,452 rows
- Labels: negative 9087 (~63%), neutral 3067 (~21%), positive 2298 (~16%)
- The data is imbalanced, so accuracy alone is misleading. A model that always says "negative" gets ~63% accuracy. I use macro-F1 instead.

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
- ngram_range (1, 2) -> (1, 1): macro-F1 changed from 0.73 to 0.71
- class_weight="balanced" removed: recall dropped most for positive class

## Why 80/20 split?
An 80/20 split strikes an ideal balance between training data volume and evaluation reliability. Using 80% of the data provides the model with enough examples to capture nuanced linguistic patterns and sentiment cues, while reserving 20% (nearly 2,900 independent samples) ensures a robust, unbiased test set to accurately measure generalization performance without overfitting.
