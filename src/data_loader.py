import numpy as np
from src.feature_extraction import extract_features

def load_word_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return set(w.strip().lower() for w in f.readlines())

def load_reviews(path, label, pos_words, neg_words):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    X = [extract_features(line, pos_words, neg_words) for line in lines]
    y = [label] * len(X)
    return np.array(X), np.array(y)
