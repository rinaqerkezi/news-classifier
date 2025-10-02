from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("../models/news_model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    text_vec = vectorizer.transform([text])
    pred = model.predict(text_vec)[0]
    return jsonify({"category": int(pred)})

if __name__ == "__main__":
    app.run(debug=True)