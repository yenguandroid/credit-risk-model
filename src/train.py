
# src/train.py

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# -------------------------------------------------
# Load Data
# -------------------------------------------------

df = pd.read_csv(
    "data/processed/customer_with_target.csv"
)

X = df.drop(
    columns=[
        "CustomerId",
        "is_high_risk"
    ]
)

X = pd.get_dummies(X, drop_first=True)

y = df["is_high_risk"]


# -------------------------------------------------
# Train Test Split
# -------------------------------------------------

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
)


# -------------------------------------------------
# Evaluation Helper
# -------------------------------------------------

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":
            accuracy_score(
                y_test,
                y_pred
            ),

        "precision":
            precision_score(
                y_test,
                y_pred
            ),

        "recall":
            recall_score(
                y_test,
                y_pred
            ),

        "f1_score":
            f1_score(
                y_test,
                y_pred
            ),

        "roc_auc":
            roc_auc_score(
                y_test,
                y_prob
            )
    }

    return metrics


# -------------------------------------------------
# Logistic Regression
# -------------------------------------------------

with mlflow.start_run(
    run_name="LogisticRegression"
):

    logistic = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    logistic.fit(
        X_train,
        y_train
    )

    metrics = evaluate_model(
        logistic,
        X_test,
        y_test
    )

    mlflow.log_params({
        "model": "LogisticRegression",
        "max_iter": 1000
    })

    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(
        logistic,
        "model"
    )

    print(metrics)


# -------------------------------------------------
# Random Forest Grid Search
# -------------------------------------------------

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, 15]
}

grid_search = GridSearchCV(
    RandomForestClassifier(
        random_state=42
    ),
    param_grid,
    cv=5,
    scoring="roc_auc"
)

grid_search.fit(
    X_train,
    y_train
)

best_rf = (
    grid_search.best_estimator_
)

rf_metrics = evaluate_model(
    best_rf,
    X_test,
    y_test
)


# -------------------------------------------------
# MLflow Logging
# -------------------------------------------------

with mlflow.start_run(
    run_name="RandomForest"
):

    mlflow.log_params(
        grid_search.best_params_
    )

    mlflow.log_metrics(
        rf_metrics
    )

    mlflow.sklearn.log_model(
        best_rf,
        "model"
    )

    print(rf_metrics)

print(
    "Best RF Parameters:",
    grid_search.best_params_
)

