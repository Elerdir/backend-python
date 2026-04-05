FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV APP_DATA_DIR=/app/data
ENV HF_HOME=/app/data/hf-cache
ENV HF_HUB_CACHE=/app/data/hf-cache/hub
ENV DEFAULT_MODEL_ID=runwayml/stable-diffusion-v1-5

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]