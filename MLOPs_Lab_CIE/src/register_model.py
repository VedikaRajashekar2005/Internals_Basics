import mlflow
import mlflow.sklearn
import json
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

mlflow.set_experiment("pipelineiq-build-time-min")

# Load data and model
df = pd.read_csv("data/training_data.csv")
X = df[["test_count", "artifact_size_mb", "parallel_jobs", "is_full_rebuild"]]
y = df["build_time_min"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

preds = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

# Log and register
with mlflow.start_run(run_name="register-best-model") as run:
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="pipelineiq-build-time-min-predictor"
    )
    run_id = run.info.run_id

# Get version number
client = mlflow.tracking.MlflowClient()
versions = client.get_latest_versions("pipelineiq-build-time-min-predictor")
version_number = int(versions[0].version)

output = {
    "registered_model_name": "pipelineiq-build-time-min-predictor",
    "version": version_number,
    "run_id": run_id,
    "source_metric": "rmse",
    "source_metric_value": round(rmse, 4)
}

os.makedirs("results", exist_ok=True)
with open("results/step4_s6.json", "w") as f:
    json.dump(output, f, indent=2)

print("Task 4 done. Version:", version_number, "Run ID:", run_id)
print("Saved: results/step4_s6.json")