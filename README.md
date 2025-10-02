# news-classifier

A simple machine learning project that classifies news articles into categories using NLP and Naive Bayes classifier.

Features: 
- Text Preprocessing: Lowercasing, stopword removal, punctuation removal
- Machine Learning model : Naive Bayes with TF-IDF vectorization
- User-Friendly Flask web application
- Interactive CLI
- Model Training

Performance:
- Accuracy: 89.4%
- Categories : 4 news categories (The class ids are numbered 1-4 where 1 represents World, 2 represents Sports, 3 represents Business and 4 represents Sci/Tech.)
- Test Data: 24,000 news articles
- Training Data: 120,000 news articles



### Python 3.8+
- pip
 ### Installation

 1. Clone the repository
 ```bash
  git clone <your-repository-url>
   cd news-classifier

2. python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


3.pip install flask pandas scikit-learn nltk joblib seaborn matplotlib

4.WEB Interface:
python src/app.py
Then open http://127.0.0.1:5000 in your browser.

- if you want to test the model:
python src/predict.py


-if you want to retrain:
python src/train.py


link for the news articles dataset :

https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset?resource=download&select=train.csv 
https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset?resource=download&select=test.csv

