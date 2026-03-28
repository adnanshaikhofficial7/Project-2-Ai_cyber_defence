from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("url_vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    prob = None
    url = ""

    if request.method == "POST":
        url = request.form["url"]
        X = vectorizer.transform([url])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0].max()

        result = "Phishing" if pred == 1 else "Legitimate"
        prob = round(proba * 100, 2)

    return render_template("index.html", result=result, prob=prob, url=url)

if __name__ == "__main__":
    app.run(debug=True)