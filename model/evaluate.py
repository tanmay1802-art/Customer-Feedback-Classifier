import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

LABELS = ["negative", "neutral", "positive"]

# Same split as train.py, so we get the exact same test set
df = pd.read_csv("data/clean.csv")
_, X_test, _, y_test = train_test_split(
    df["text"], df["airline_sentiment"],
    test_size=0.2, random_state=42, stratify=df["airline_sentiment"],
)

model = joblib.load("model/sentiment_model.joblib")
pred = model.predict(X_test)

# 1. Baseline: always predict "negative"
base_pred = ["negative"] * len(y_test)
print("=== Baseline (always negative) ===")
print(f"accuracy: {accuracy_score(y_test, base_pred):.3f}")
print(f"macro-F1: {f1_score(y_test, base_pred, average='macro', zero_division=0):.3f}")

print("\n=== Model (logreg) ===")
print(f"accuracy: {accuracy_score(y_test, pred):.3f}")
print(f"macro-F1: {f1_score(y_test, pred, average='macro'):.3f}")

# 2. Confusion matrix (rows = true label, columns = predicted label)
cm = confusion_matrix(y_test, pred, labels=LABELS)
cm_df = pd.DataFrame(
    cm,
    index=[f"true_{l}" for l in LABELS],
    columns=[f"pred_{l}" for l in LABELS],
)
print("\n=== Confusion matrix (counts) ===")
print(cm_df)

print("\n=== Confusion matrix (each row sums to 1, diagonal = recall) ===")
print(cm_df.div(cm_df.sum(axis=1), axis=0).round(2))

# 3. 15 misclassified tweets, with the model's confidence
proba = model.predict_proba(X_test)
results = pd.DataFrame({
    "text": X_test.values,
    "true": y_test.values,
    "pred": pred,
    "confidence": proba.max(axis=1),
})
wrong = results[results["true"] != results["pred"]]
print(f"\n=== Misclassified: {len(wrong)} of {len(results)} ===")
for _, r in wrong.sample(15, random_state=42).iterrows():
    print(f"\ntrue={r['true']} | pred={r['pred']} | conf={r['confidence']:.2f}")
    print(r["text"])
