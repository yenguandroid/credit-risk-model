import mlflow

mlflow.set_experiment("test_experiment")

with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", 0.95)

print("Run logged successfully!")