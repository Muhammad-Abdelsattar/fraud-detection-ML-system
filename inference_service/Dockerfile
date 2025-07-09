FROM python:3.10-slim

WORKDIR /app


RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


<<<<<<< HEAD:inference_service/Dockerfile
COPY ./app ./app



EXPOSE 8000


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
=======
# Define default environment variables
# These can be overridden at runtime (e.g., docker run -e DATA_PATH=/custom/data.csv ...)
ENV DATA_PATH="/teamspace/studios/this_studio/data_generation/final_project/synthetic_fraud_data_v3.csv"
ENV ARTIFACTS_DIR="/fraud_detection/artifacts" # Defaulting to a path inside the container
ENV TEST_SIZE_RATIO="0.2"
ENV N_TRIALS="3"
ENV WANDB_API_KEY="" # Declare WANDB_API_KEY, to be provided at runtime

ENTRYPOINT ["python", "-m", "main"]
>>>>>>> feature/gcs-local-paths:training_pipeline/Dockerfile
