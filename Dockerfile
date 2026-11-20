FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fetch_model.py app.py ./

ARG MLFLOW_TRACKING_URI
ARG MODEL_NAME=my-model

# Fetch the latest registered model from DagsHub (via MLflow) at build time.
# The token is passed as a BuildKit secret, so it is never stored in the image.
RUN --mount=type=secret,id=dagshub_token \
    export MLFLOW_TRACKING_USERNAME="$(cat /run/secrets/dagshub_token)" && \
    export MLFLOW_TRACKING_PASSWORD="$(cat /run/secrets/dagshub_token)" && \
    python fetch_model.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]