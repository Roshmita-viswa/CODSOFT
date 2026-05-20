import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib


def clean_message(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Loading dataset...")
sms_data = pd.read_csv("spam.csv", encoding="latin-1")
sms_data = sms_data.loc[:, ["v1", "v2"]]
sms_data.columns = ["label", "message"]

sms_data = sms_data.dropna().reset_index(drop=True)
sms_data["message"] = sms_data["message"].astype(str).map(clean_message)
sms_data = sms_data[sms_data["message"].str.len() > 0]
sms_data = sms_data.drop_duplicates(subset="message", keep="first").reset_index(drop=True)

print("\nFirst 5 rows:")
print(sms_data.head())

print("\nLabel distribution:")
print(sms_data["label"].value_counts())

sms_data["label_num"] = sms_data["label"].map({"ham": 0, "spam": 1})

X = sms_data["message"]
y = sms_data["label_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel saved as spam_model.pkl")
print("Vectorizer saved as vectorizer.pkl")
