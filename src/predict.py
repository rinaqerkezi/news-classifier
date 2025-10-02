import joblib

model=joblib.load("../models/news_model.pkl")
vectorizer=joblib.load("../models/vectorizer.pkl")

def predict_category(text):
    text_vec=vectorizer.transform([text])
    pred=model.predict(text_vec)[0]
    return pred

if __name__ == "__main__":
    while True:
        user_input=input("Shkruaj tekstin/Write the text:")
        print("Kategorija e parashikuar/Predicted Category:", predict_category(user_input))