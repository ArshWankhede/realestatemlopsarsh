import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def retrain_model():
    print("Loading data for retraining...")
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    # Use the EXACT same experiment name so they are grouped together
    mlflow.set_experiment("Real_Estate_Price_Prediction")

    with mlflow.start_run():
        print("Training NEW, better model...")
        
        # NEW HYPERPARAMETERS
        n_estimators = 100
        max_depth = 10
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        
        mlflow.log_metric("mse", mse)
        print(f"New model trained with MSE: {mse}")

        mlflow.sklearn.log_model(model, "random_forest_model")
        print("New model saved to MLflow!")

if __name__ == "__main__":
    retrain_model()