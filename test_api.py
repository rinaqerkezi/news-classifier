import requests
import json

url = "http://127.0.0.1:5000/predict"


test_texts = [
    "Stock markets reached record highs today as investors cheered positive economic data",
    "Scientists discovered a new species of dinosaur in Argentina",
    "The football team won the championship with a last-minute goal",
    "New smartphone released with advanced AI features"
]

for text in test_texts:
    data = {"text": text}
    response = requests.post(url, json=data)
    print(f"Text: {text}")
    print(f"Predicted category: {response.json()}")
    print("-" * 50)