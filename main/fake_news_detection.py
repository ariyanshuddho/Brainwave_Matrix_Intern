import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Load datasets (notice ../data)
df_true = pd.read_csv("../data/true.csv")
df_fake = pd.read_csv("../data/fake.csv")

# Find text column
def find_text_column(df):
    for col in df.columns:
        if any(word in col.lower() for word in ["text", "content", "article", "headline", "title"]):
            return col
    raise ValueError("No text column found")

text_col_true = find_text_column(df_true)
text_col_fake = find_text_column(df_fake)

# Merge and label
df_t = df_true[[text_col_true]].rename(columns={text_col_true: "text"})
df_t["label"] = 0
df_f = df_fake[[text_col_fake]].rename(columns={text_col_fake: "text"})
df_f["label"] = 1
df = pd.concat([df_t, df_f], ignore_index=True).sample(frac=1, random_state=42)
df["text"] = df["text"].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.15, stratify=df["label"], random_state=42)

# TF-IDF
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = lr.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model + vectorizer
pickle.dump(tfidf, open("../data/tfidf_vectorizer.pkl", "wb"))
pickle.dump(lr, open("../data/logistic_regression_model.pkl", "wb"))
