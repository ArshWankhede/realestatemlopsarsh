import requests
import random

url = "http://127.0.0.1:8000/predict"

for _ in range(20):
    payload = {
        "MedInc": round(random.uniform(1.5, 15.0), 2),
        "HouseAge": random.randint(1, 52),
        "AveRooms": round(random.uniform(3.0, 10.0), 1),
        "AveBedrms": round(random.uniform(0.8, 1.5), 1),
        "Population": random.randint(100, 5000),
        "AveOccup": round(random.uniform(2.0, 5.0), 1),
        "Latitude": round(random.uniform(32.5, 42.0), 2),
        "Longitude": round(random.uniform(-124.3, -114.3), 2)
    }
    response = requests.post(url, json=payload)
    print(f"Status: {response.status_code}, Predicted: {response.json()['predicted_price_in_100k']}")