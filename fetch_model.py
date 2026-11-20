import os
import shutil

import mlflow
from mlflow.tracking import MlflowClient

# MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD
# are provided as environment variables during the Docker build.
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
model_name = os.environ.get("MODEL_NAME", "my-model")

# Find the latest registered version of the model
client = MlflowClient()
versions = client.search_model_versions(f"name='{model_name}'")
if not versions:
    raise SystemExit(f"No registered model named '{model_name}' found")
latest = max(versions, key=lambda v: int(v.version))
print(f"Latest version of '{model_name}': {latest.version}")

# Download it
mlflow.artifacts.download_artifacts(
    artifact_uri=f"models:/{model_name}/{latest.version}",
    dst_path="model_download",
)

# Locate the folder that contains the MLmodel file and move it to ./model
found = None
for root, _dirs, files in os.walk("model_download"):
    if "MLmodel" in files:
        found = root
        break
if found is None:
    raise SystemExit("MLmodel file not found in downloaded artifacts")

shutil.rmtree("model", ignore_errors=True)
shutil.move(found, "model")
shutil.rmtree("model_download", ignore_errors=True)
print("Model ready in ./model")