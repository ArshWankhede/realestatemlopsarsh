import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_model():
    print("Loading data...")
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    # 1. Set our experiment name in MLflow
    mlflow.set_experiment("Real_Estate_Price_Prediction")

    # 2. Start the MLflow run
    with mlflow.start_run():
        print("Training model...")
        
        # Define hyperparameters
        n_estimators = 50
        max_depth = 5
        
        # Log parameters to MLflow so we remember what we used later
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Train the model
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        
        # Log the performance metric
        mlflow.log_metric("mse", mse)
        print(f"Model trained with MSE: {mse}")

        # 3. Log the model artifact itself
        mlflow.sklearn.log_model(model, "random_forest_model")
        print("Model saved to MLflow!")

if __name__ == "__main__":
    train_model()