import numpy as np
import joblib
from src.data_loader import load_word_list
from src.feature_extraction import extract_features

pos_words = load_word_list("data/positive-words.txt")
neg_words = load_word_list("data/negative-words.txt")

model = joblib.load("output/best_model.pkl")

with open("data/challenge_data.txt", "r", encoding="utf-8") as f:
    reviews = f.readlines()

X = np.array([
    extract_features(r, pos_words, neg_words) for r in reviews
])

preds = model.predict(X)

output = "".join(str(int(p)) for p in preds)

assert len(output) == 5000

with open("output/submission.txt", "w", encoding="utf-8") as f:
    f.write(output)

print("submission.txt generated successfully")
