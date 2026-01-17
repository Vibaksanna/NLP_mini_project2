# Mini Project 2 – Text Classification (NLP)

This project implements a sentiment classification system for product reviews.  
Each review is classified as **positive (1)** or **negative (0)** using multiple machine learning models and feature extraction techniques.

The project strictly follows the assignment requirements and includes:

- Correct **80% / 20% data split**
- Multiple models for comparison
- Accuracy-based evaluation
- A blind challenge dataset
- Proper output formatting

---

## 📁 Project Structure

```
Text_Classification/
│
├── data/
│   ├── positive-reviews.txt
│   ├── negative-reviews.txt
│   ├── positive-words.txt
│   ├── negative-words.txt
│   └── challenge_data.txt
│
├── output/
│   ├── best_model.pkl
│   ├── tfidf_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── submission.txt
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_extraction.py
│   ├── models.py
│   ├── train.py
│   ├── train_tfidf.py
│   ├── predict_challenge.py
│   └── predict_challenge_tfidf.py
│
├── venv/
├── requirements.txt
└── README.md
```

---

## 🧠 Models Implemented

### Hand-Crafted Feature Models

- Logistic Regression
- Naive Bayes
- Random Forest

### TF-IDF Feature Model

- Logistic Regression (**best performing**)

---

## 📊 Accuracy Results

| Model                   | Features     | Accuracy   |
| ----------------------- | ------------ | ---------- |
| Naive Bayes             | Hand-crafted | 80.53%     |
| Logistic Regression     | Hand-crafted | 82.53%     |
| Random Forest           | Hand-crafted | 82.57%     |
| **Logistic Regression** | **TF-IDF**   | **91.31%** |

The TF-IDF + Logistic Regression model achieved the highest accuracy and was selected for the final challenge prediction.

---

## ⚙️ Setup Instructions

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### 🔹 Train hand-crafted feature models

```bash
python -m src.train
```

This will:

- Train Logistic Regression, Naive Bayes, and Random Forest
- Evaluate accuracy
- Save the best model to `output/best_model.pkl`

---

### 🔹 Train TF-IDF model (Recommended)

```bash
python -m src.train_tfidf
```

This will:

- Train TF-IDF + Logistic Regression
- Print accuracy
- Save:
  - `output/tfidf_model.pkl`
  - `output/tfidf_vectorizer.pkl`

---

### 🔹 Generate Challenge Predictions (FINAL)

```bash
python -m src.predict_challenge_tfidf
```

This will generate:

```
output/submission.txt
```

✔ Exactly **5000 characters**  
✔ No spaces  
✔ No new lines  
✔ `0 = negative`, `1 = positive`

---

## 📌 Important Notes

- The **challenge_data.txt** file is **NOT used for training**
- Only the labeled datasets are split into training and testing sets
- Data splitting follows the **top 80% / bottom 20% rule**
- Accuracy is the only evaluation metric used
- The TF-IDF model is recommended for submission

---

## 🧾 Final Submission

Submit:

- `output/submission.txt`
- Source code (`src/`)
- Report PDF

---

## 🏁 Conclusion

This project demonstrates the importance of feature representation in sentiment analysis.  
While hand-crafted features provide reasonable performance, TF-IDF representations combined with Logistic Regression significantly improve accuracy and generalization.

---

**Author:**  
Mini Project 2 – NLP Text Classification
