import joblib
import os
from config import get_category_name 


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
models_dir = os.path.join(project_root, 'models')

model_path = os.path.join(models_dir, 'news_model.pkl')
vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')

print(f"Looking for model at: {model_path}")

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model and vectorizer loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Please make sure you've run train.py first to create the model files.")
    exit(1)

def predict_category(text):
    text_vec = vectorizer.transform([text])
    pred = model.predict(text_vec)[0]
    return pred

if __name__ == "__main__":
    print("News Classifier - Shkruaj tekstin e lajmit / Write news text:")
    print("Categories: World (1), Sports (2), Business (3), Sci/Tech (4)")
    print("Press Ctrl+C to exit")
    while True:
        try:
            user_input = input("\nShkruaj tekstin/Write the text: ")
            if user_input.lower() in ['exit', 'quit', '']:
                break
            category_numeric = predict_category(user_input)
            category_info = get_category_name(category_numeric)
            print(f"Kategorija e parashikuar/Predicted Category: {category_info['name']} (ID: {category_info['id']})")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")