import requests
import json
import os

BASE = "http://localhost:8500"

health = requests.get(f"{BASE}/heartbeat").json()

test_input = {
    "test_count": 392,
    "artifact_size_mb": 164,
    "parallel_jobs": 5,
    "is_full_rebuild": 1
}

pred_resp = requests.post(f"{BASE}/predict", json=test_input).json()
prediction = pred_resp["prediction"]

output = {
    "health_endpoint": "/heartbeat",
    "predict_endpoint": "/predict",
    "port": 8500,
    "health_response": health,
    "test_input": test_input,
    "prediction": prediction
}

os.makedirs("results", exist_ok=True)
with open("results/step3_s4.json", "w") as f:
    json.dump(output, f, indent=2)

print("Task 3 done. Prediction:", prediction)
print("Saved: results/step3_s4.json")