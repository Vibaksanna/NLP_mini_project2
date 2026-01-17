import math
import re
from textblob import TextBlob

def extract_features(text, pos_words, neg_words):
    text_lower = text.lower()
    words = re.findall(r"\b\w+\b", text_lower)

    pos_count = sum(1 for w in words if w in pos_words)
    neg_count = sum(1 for w in words if w in neg_words)

    has_no = 1 if "no" in words else 0
    pronouns = {"i", "me", "my", "you", "your"}
    pronoun_count = sum(1 for w in words if w in pronouns)
    has_exclaim = 1 if "!" in text else 0
    log_length = math.log(len(words) + 1)

    # Extra features
    word_count = len(words)
    uppercase_count = sum(1 for w in text.split() if w.isupper())
    polarity = TextBlob(text).sentiment.polarity
    pos_ratio = pos_count / (word_count + 1)
    neg_ratio = neg_count / (word_count + 1)

    return [
        pos_count,
        neg_count,
        has_no,
        pronoun_count,
        has_exclaim,
        log_length,
        word_count,
        uppercase_count,
        polarity,
        pos_ratio,
        neg_ratio
    ]
