FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Only the API runtime dependencies: training-only packages (matplotlib,
# seaborn, jupyter) stay out of the image.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Code plus the model artifacts the service loads at startup.
COPY data.py serve.py ./
COPY ANN_BinaryVLE/ann_vle_model.h5 ANN_BinaryVLE/preprocessing_pipeline.pkl \
     ANN_BinaryVLE/metrics.json ./ANN_BinaryVLE/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
