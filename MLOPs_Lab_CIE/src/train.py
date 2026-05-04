import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
df = pd.read_csv("data/training_data.csv")
X = df[["test_count", "artifact_size_mb", "parallel_jobs", "is_full_rebuild"]]
y = df["build_time_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("pipelineiq-build-time-min")

results = []

# --- Ridge ---
with mlflow.start_run(run_name="Ridge"):
    mlflow.set_tag("team", "ml_engineering")
    alpha = 1.0
    mlflow.log_param("alpha", alpha)

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model")

    results.append({"name": "Ridge", "mae": round(mae, 4), "rmse": round(rmse, 4)})
    print(f"Ridge  — MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# --- RandomForest ---
with mlflow.start_run(run_name="RandomForest"):
    mlflow.set_tag("team", "ml_engineering")
    n_estimators = 100
    max_depth = None
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model")

    results.append({"name": "RandomForest", "mae": round(mae, 4), "rmse": round(rmse, 4)})
    print(f"RandomForest — MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Pick best by RMSE (lower is better)
best = min(results, key=lambda x: x["rmse"])

output = {
    "experiment_name": "pipelineiq-build-time-min",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}

os.makedirs("results", exist_ok=True)
with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nTask 1 done. Best model:", best["name"])
print("Saved: results/step1_s1.json")