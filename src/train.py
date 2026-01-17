import os
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from src.models import get_models
from src.data_loader import load_word_list
from src.feature_extraction import extract_features

# ======================================================
# Load sentiment word lists
# ======================================================
pos_words = load_word_list("data/positive-words.txt")
neg_words = load_word_list("data/negative-words.txt")

# ======================================================
# Helper function: TOP 80% split (PDF requirement)
# ======================================================
def split_top_80(file_path, label):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) == 0:
        raise ValueError(f"{file_path} is empty")

    split_index = int(0.8 * len(lines))

    train_lines = lines[:split_index]
    test_lines = lines[split_index:]

    X_train = [extract_features(line, pos_words, neg_words) for line in train_lines]
    y_train = [label] * len(X_train)

    X_test = [extract_features(line, pos_words, neg_words) for line in test_lines]
    y_test = [label] * len(X_test)

    return X_train, y_train, X_test, y_test

# ======================================================
# Load POSITIVE reviews
# ======================================================
X_train_pos, y_train_pos, X_test_pos, y_test_pos = split_top_80(
    "data/positive-reviews.txt", 1
)

# ======================================================
# Load NEGATIVE reviews
# ======================================================
X_train_neg, y_train_neg, X_test_neg, y_test_neg = split_top_80(
    "data/negative-reviews.txt", 0
)

# ======================================================
# Debug counts (safety check)
# ======================================================
print(f"Positive train samples: {len(X_train_pos)}")
print(f"Negative train samples: {len(X_train_neg)}")
print(f"Positive test samples: {len(X_test_pos)}")
print(f"Negative test samples: {len(X_test_neg)}")

# ======================================================
# Combine datasets
# ======================================================
X_train = np.array(X_train_pos + X_train_neg)
y_train = np.array(y_train_pos + y_train_neg)

X_test = np.array(X_test_pos + X_test_neg)
y_test = np.array(y_test_pos + y_test_neg)

# ======================================================
# Shuffle (CRITICAL for model stability)
# ======================================================
X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_test, y_test = shuffle(X_test, y_test, random_state=42)

# ======================================================
# Train & evaluate models
# ======================================================
models = get_models()

best_model = None
best_accuracy = 0.0

print("\nModel accuracy results:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"{name}: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print(f"\nBest model selected: {best_model.__class__.__name__}")

# ======================================================
# Save best model
# ======================================================
os.makedirs("output", exist_ok=True)
joblib.dump(best_model, "output/best_model.pkl")

print("Best model saved to output/best_model.pkl")
