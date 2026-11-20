# train.py
# Plain-Python version of MLFlowSetup.ipynb (with fixes).
#
# Install once (in PowerShell):
#   python -m pip install scikit-learn "mlflow<3" dagshub
#
# Run (from inside the my-dvc-project folder):
#   python train.py

import dagshub
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# --- Connect to DagsHub (a browser window opens the first time to log in) ---
dagshub.init(repo_owner="ryn2004t", repo_name="my-dvc-project", mlflow=True)

# --- Experiment name (created automatically if it does not exist) ---
mlflow.set_experiment("RandomForest_V1.0.1")

# --- Load data ---
data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)


def train_and_log_model(n_estimators, max_depth):
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)

        # registered_model_name puts the model in the MLflow Model Registry,
        # which is what the Dockerfile fetches later (models:/my-model/latest)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="my-model",
        )

        print(
            f"Logged model: n_estimators={n_estimators}, "
            f"max_depth={max_depth}, accuracy={accuracy:.4f}"
        )


if __name__ == "__main__":
    print("Training and logging Model 1...")
    train_and_log_model(n_estimators=20, max_depth=5)

    print("\nTraining and logging Model 2...")
    train_and_log_model(n_estimators=100, max_depth=10)

    print("\nDone. Check the Experiments tab on your DagsHub repo.")