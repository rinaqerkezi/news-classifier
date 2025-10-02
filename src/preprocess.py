import pandas as pd
import nltk
from ntlk.corpus import stopwords
import re

ntlk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def load_and_clean_data(filepath):
    df=pd.read_csv(filepath, header=None)
    df.columns=['ClassIndex','Title', 'Description']
    df['Text'] = df['Title'] + " " + df['Description']
    df['Text'] = df['Text'].apply(clean_text)
    df['Category'] = df['ClassIndex'] - 1
    return df[['Text', 'Category']]

if __name__ == "__main__":
    df_train = load_and_clean_data("../data/train.csv")
    print(df_train.head())