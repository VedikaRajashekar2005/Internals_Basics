import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer

# Load data
df = pd.read_csv("data/training_data.csv")
X = df[["test_count", "artifact_size_mb", "parallel_jobs", "is_full_rebuild"]]
y = df["build_time_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [50, 150, 250],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 3, 5],
}

total_trials = (
    len(param_grid["n_estimators"])
    * len(param_grid["max_depth"])
    * len(param_grid["min_samples_split"])
)

mlflow.set_experiment("pipelineiq-build-time-min")

neg_mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

with mlflow.start_run(run_name="tuning-pipelineiq") as parent_run:
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
    )
    grid.fit(X_train, y_train)

    # Log each trial as a nested run
    for i, params in enumerate(grid.cv_results_["params"]):
        with mlflow.start_run(run_name=f"trial_{i+1}", nested=True):
            mlflow.log_params(params)
            rmse_cv = np.sqrt(-grid.cv_results_["mean_test_score"][i])
            mlflow.log_metric("cv_rmse", rmse_cv)

    best_params = grid.best_params_
    best_model = grid.best_estimator_

    # Evaluate on test set
    preds = best_model.predict(X_test)
    best_mae = mean_absolute_error(y_test, preds)
    best_rmse = np.sqrt(mean_squared_error(y_test, preds))

    # CV MAE (re-run best estimator with CV scoring for MAE)
    from sklearn.model_selection import cross_val_score
    cv_mae_scores = cross_val_score(
        RandomForestRegressor(**best_params, random_state=42),
        X_train, y_train,
        cv=3,
        scoring="neg_mean_absolute_error"
    )
    best_cv_mae = round(-cv_mae_scores.mean(), 4)

    mlflow.log_params(best_params)
    mlflow.log_metric("best_mae", best_mae)
    mlflow.log_metric("best_rmse", best_rmse)
    mlflow.log_metric("best_cv_mae", best_cv_mae)
    mlflow.sklearn.log_model(best_model, "tuned_model")

    # Save model to disk for Task 3
    os.makedirs("models", exist_ok=True)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    output = {
        "search_type": "grid",
        "n_folds": 3,
        "total_trials": total_trials,
        "best_params": best_params,
        "best_mae": round(best_mae, 4),
        "best_cv_mae": best_cv_mae,
        "parent_run_name": "tuning-pipelineiq"
    }

    os.makedirs("results", exist_ok=True)
    with open("results/step2_s2.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Task 2 done. Best params:", best_params)
    print("Saved: results/step2_s2.json")
    print("Saved: models/best_model.pkl")