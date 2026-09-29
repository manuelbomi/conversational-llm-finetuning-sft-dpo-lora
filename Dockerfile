# Serving image only -- training is meant to be run on a GPU host/notebook/
# job runner, not inside this container. See README.md "Setup" for the
# training-side install sequence (torch from the CPU or CUDA index first).
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src

ENV BASE_MODEL_NAME=sshleifer/tiny-gpt2
ENV DEFAULT_ADAPTER_DIR=/app/artifacts/dpo-adapter

EXPOSE 8080

CMD ["uvicorn", "src.serve.main:app", "--host", "0.0.0.0", "--port", "8080"]
