import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import load_and_clean_data
from collections import Counter


df = load_and_clean_data("../data/train.csv")


sns.countplot(x='Category', data=df)
plt.title("Number of samples per category")
plt.show()

all_words = ' '.join(df['Text']).split()
most_common = Counter(all_words).most_common(20)
print("Most frequent words:", most_common)


X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Category'], test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

joblib.dump(model, '../models/news_model.pkl')
joblib.dump(vectorizer, '../models/vectorizer.pkl')
