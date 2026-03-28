import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset (CSV with columns: url,label)
data = pd.read_csv("phishing_urls.csv")  # label: 1=phishing, 0=legit

X = data["url"].astype(str)
y = data["label"].astype(int)

vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5))
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

joblib.dump(model, "phishing_model.pkl")
joblib.dump(vectorizer, "url_vectorizer.pkl")

print("Model saved: phishing_model.pkl")
print("Vectorizer saved: url_vectorizer.pkl")