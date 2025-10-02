from flask import Flask, request, jsonify
import joblib
import os
from config import get_category_name  

app = Flask(__name__)


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
models_dir = os.path.join(project_root, 'models')

model_path = os.path.join(models_dir, 'news_model.pkl')
vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>News Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            textarea { width: 100%; height: 150px; margin: 10px 0; padding: 10px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            .result { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }
            .category { font-weight: bold; color: #007bff; }
        </style>
    </head>
    <body>
        <h1>📰 News Classification</h1>
        <p><strong>Categories:</strong> World (1), Sports (2), Business (3), Sci/Tech (4)</p>
        <form id="classifyForm">
            <textarea id="textInput" placeholder="Enter news text here..."></textarea>
            <br>
            <button type="submit">Classify News</button>
        </form>
        <div id="result"></div>
        
        <script>
            document.getElementById('classifyForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const text = document.getElementById('textInput').value;
                const resultDiv = document.getElementById('result');
                
                if (!text) {
                    resultDiv.innerHTML = '<div class="result">Please enter some text</div>';
                    return;
                }
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({text: text})
                    });
                    
                    const data = await response.json();
                    resultDiv.innerHTML = `<div class="result">
                        <strong>Predicted Category:</strong> <span class="category">${data.category_name} (ID: ${data.category_id})</span><br>
                        <strong>Text:</strong> ${text.substring(0, 100)}...
                    </div>`;
                } catch (error) {
                    resultDiv.innerHTML = `<div class="result">Error: ${error}</div>`;
                }
            });
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    text_vec = vectorizer.transform([text])
    pred = model.predict(text_vec)[0]
    
   
    category_info = get_category_name(pred)
    
    return jsonify({
        "category_id": category_info["id"],
        "category_name": category_info["name"],
        "category_numeric": int(pred)
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)