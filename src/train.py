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
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  

models_dir = os.path.join(project_root, 'models')
outputs_dir = os.path.join(project_root, 'outputs')
data_dir = os.path.join(project_root, 'data')

print(f"Current directory (src): {current_dir}")
print(f"Project root: {project_root}")
print(f"Models directory: {models_dir}")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

df_train = load_and_clean_data(os.path.join(data_dir, 'train.csv'))


plt.figure(figsize=(10, 6))
sns.countplot(x='Category', data=df_train)
plt.title("Numri i mostrave per kategori/number of samples:")
plt.savefig(os.path.join(outputs_dir, 'category_distribution.png'))
plt.show()

all_words = ' '.join(df_train['Text']).split()
most_common = Counter(all_words).most_common(10)
print("Fjalet qe u perdoren me se shpeshti/Most frequent word:", most_common)

X_train, X_test, y_train, y_test = train_test_split(df_train['Text'], df_train['Category'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig(os.path.join(outputs_dir, 'confusion_matrix.png'))
plt.show()


model_path = os.path.join(models_dir, 'news_model.pkl')
vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"Model saved to: {model_path}")
print(f"Vectorizer saved to: {vectorizer_path}")

if os.path.exists(model_path):
    print("Model file verified")
    print(f"File size: {os.path.getsize(model_path)} bytes")
else:
    print("Model file not found!")

if os.path.exists(vectorizer_path):
    print("Vectorizer file verified")
    print(f"File size: {os.path.getsize(vectorizer_path)} bytes")
else:
    print("Vectorizer file not found!")

print("Training completed successfully!")