import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

df = pd.read_csv("data/clean.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["airline_sentiment"],
    test_size=0.2, random_state=42, stratify=df["airline_sentiment"],
)

def make(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", clf),
    ])

models = {
    "logreg": make(LogisticRegression(max_iter=1000, class_weight="balanced")),
    "linearsvc": make(LinearSVC(class_weight="balanced")),
}

for name, m in models.items():
    m.fit(X_train, y_train)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, m.predict(X_test)))

joblib.dump(models["logreg"], "model/sentiment_model.joblib")
