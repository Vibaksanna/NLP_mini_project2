import joblib

# Load model & vectorizer
model = joblib.load("output/tfidf_model.pkl")
vectorizer = joblib.load("output/tfidf_vectorizer.pkl")

# Load challenge data
with open("data/challenge_data.txt", "r", encoding="utf-8") as f:
    reviews = f.readlines()

X = vectorizer.transform(reviews)
preds = model.predict(X)

output = "".join(str(int(p)) for p in preds)

assert len(output) == 5000

with open("output/submission.txt", "w", encoding="utf-8") as f:
    f.write(output)

print("submission.txt generated using TF-IDF model")
