import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def load_and_clean_data(filepath):
    
    df = pd.read_csv(filepath)
    
   
    df['Text'] = df['Title'] + " " + df['Description']
    df['Text'] = df['Text'].apply(clean_text)
    
   
    df['Category'] = df['Class Index'].astype(int) - 1
    
    return df[['Text', 'Category']]

if __name__ == "__main__":
   df_train = load_and_clean_data(r"C:/Users/rinaq/OneDrive/Desktop/news-classifier/data/train.csv")
   print(df_train.head())