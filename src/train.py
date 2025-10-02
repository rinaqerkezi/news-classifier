import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import load_and_clean_data
from collections import Counter
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  
models_dir = os.path.join(project_root, 'models')
outputs_dir = os.path.join(project_root, 'outputs')
data_dir = os.path.join(project_root, 'data')

os.makedirs(models_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)


print("Loading and preprocessing data...")
df_train = load_and_clean_data(os.path.join(data_dir, 'train.csv'))

plt.figure(figsize=(10, 6))
sns.countplot(x='Category', data=df_train)
plt.title("Category Distribution")
plt.savefig(os.path.join(outputs_dir, 'category_distribution.png'))
plt.close()  

print("Category distribution plot saved.")

all_words = ' '.join(df_train['Text']).split()
most_common = Counter(all_words).most_common(10)
print("Most frequent words:", most_common)


X_train, X_test, y_train, y_test = train_test_split(
    df_train['Text'], df_train['Category'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Training data shape: {X_train_vec.shape}")
print(f"Test data shape: {X_test_vec.shape}")

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}


results = {}
best_accuracy = 0
best_model_name = None
best_model = None

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

for name, model in models.items():
    print(f"\nTraining {name}...")
    start_time = time.time()
    
   
    model.fit(X_train_vec, y_train)
    
  
    y_pred = model.predict(X_test_vec)
   
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    training_time = time.time() - start_time
   
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'f1_score': f1,
        'training_time': training_time,
        'predictions': y_pred
    }
    
    print(f"{name} Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Time:      {training_time:.2f}s")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        best_model = model

print("\n" + "="*50)
print("COMPARISON SUMMARY")
print("="*50)

print("\nModel Performance Comparison:")
print("-" * 50)
print(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'Time (s)':<10}")
print("-" * 50)
for name, result in results.items():
    print(f"{name:<20} {result['accuracy']:<10.4f} {result['f1_score']:<10.4f} {result['training_time']:<10.2f}")

print(f"\n BEST MODEL: {best_model_name} (Accuracy: {best_accuracy:.4f})")


print(f"\nDetailed classification report for {best_model_name}:")
print(classification_report(y_test, results[best_model_name]['predictions']))


plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.savefig(os.path.join(outputs_dir, 'confusion_matrix_best.png'))
plt.close()  
print("Confusion matrix saved.")


results_df = pd.DataFrame([
    {
        'Model': name,
        'Accuracy': result['accuracy'],
        'F1_Score': result['f1_score'],
        'Training_Time_Seconds': result['training_time']
    }
    for name, result in results.items()
])
results_df.to_csv(os.path.join(outputs_dir, 'model_comparison.csv'), index=False)
print(f"Comparison results saved to: {os.path.join(outputs_dir, 'model_comparison.csv')}")

plt.figure(figsize=(10, 6))
models_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in models_names]
colors = ['skyblue', 'lightcoral']

bars = plt.bar(models_names, accuracies, color=colors)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0.85, 0.92)  


for bar, accuracy in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
             f'{accuracy:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, 'model_comparison_chart.png'))
plt.close()  
print("Model comparison chart saved.")

model_path = os.path.join(models_dir, 'best_model.pkl')
vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')

joblib.dump(best_model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"\n Best model ({best_model_name}) saved to: {model_path}")
print(f" Vectorizer saved to: {vectorizer_path}")

print("\n Training and comparison completed successfully!")