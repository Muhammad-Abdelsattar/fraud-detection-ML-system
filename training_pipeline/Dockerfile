FROM python:3.10-slim

WORKDIR /fraud_detection

COPY ./requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ./fraud_detection/ ./

ENTRYPOINT ["python", "-m", "main"]
