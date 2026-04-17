from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import csv
import os
from datetime import datetime

app = FastAPI(title="Real Estate Price Predictor API")

# 1. DYNAMICALLY FIND THE BEST MODEL
print("Searching for the best model in MLflow...")
experiment_name = "Real_Estate_Price_Prediction"
client = MlflowClient()

# Get the experiment ID
experiment = client.get_experiment_by_name(experiment_name)

# Search all runs in this experiment, order them by lowest MSE, and grab the top 1
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.mse ASC"],
    max_results=1
)

best_run_id = runs[0].info.run_id
best_mse = runs[0].data.metrics["mse"]

print(f"Found Best Run ID: {best_run_id} with MSE: {best_mse}")

# Load that specific best model
model_uri = f"runs:/{best_run_id}/random_forest_model"
model = mlflow.sklearn.load_model(model_uri)


# 2. Define data format
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# 3. Prediction and Telemetry Endpoint
@app.post("/predict")
def predict_price(features: HouseFeatures):
    data = [[
        features.MedInc, features.HouseAge, features.AveRooms,
        features.AveBedrms, features.Population, features.AveOccup,
        features.Latitude, features.Longitude
    ]]
    
    prediction = model.predict(data)
    predicted_price = float(prediction[0])
    
    # Telemetry
    log_file = "prediction_logs.csv"
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude", "Predicted_Price"])
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            features.MedInc, features.HouseAge, features.AveRooms,
            features.AveBedrms, features.Population, features.AveOccup,
            features.Latitude, features.Longitude, 
            predicted_price
        ])
    
    # Return the prediction AND the run_id so users know which model answered them
    return {
        "predicted_price_in_100k": predicted_price,
        "model_used": best_run_id
    }