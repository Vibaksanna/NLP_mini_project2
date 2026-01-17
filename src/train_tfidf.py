import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# ======================================================
# Helper: load lines
# ======================================================
def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()

# ======================================================
# Load data
# ======================================================
pos_lines = load_lines("data/positive-reviews.txt")
neg_lines = load_lines("data/negative-reviews.txt")

# ======================================================
# TOP 80% SPLIT (PDF-CORRECT)
# ======================================================
split_pos = int(0.8 * len(pos_lines))
split_neg = int(0.8 * len(neg_lines))

X_train_text = pos_lines[:split_pos] + neg_lines[:split_neg]
y_train = [1]*split_pos + [0]*split_neg

X_test_text = pos_lines[split_pos:] + neg_lines[split_neg:]
y_test = [1]*(len(pos_lines)-split_pos) + [0]*(len(neg_lines)-split_neg)

# Shuffle (important)
X_train_text, y_train = shuffle(X_train_text, y_train, random_state=42)
X_test_text, y_test = shuffle(X_test_text, y_test, random_state=42)

# ======================================================
# TF-IDF FEATURE EXTRACTION
# ======================================================
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# ======================================================
# MODEL (Logistic Regression)
# ======================================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print(f"TF-IDF + Logistic Regression Accuracy: {accuracy:.4f}")

# ======================================================
# SAVE MODEL + VECTORIZER
# ======================================================
os.makedirs("output", exist_ok=True)

joblib.dump(model, "output/tfidf_model.pkl")
joblib.dump(vectorizer, "output/tfidf_vectorizer.pkl")

print("TF-IDF model and vectorizer saved")
