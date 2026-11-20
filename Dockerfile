FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime dependencies only: the service executes the exported ONNX graph with
# numpy-only scaling, so TensorFlow, scikit-learn, pandas and the training plots
# stay out of the image.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Code plus the two artifacts the service loads at startup: the ONNX graph and
# the model contract (units, trained ranges, scaler, metrics).
COPY serve.py ./
COPY ANN_BinaryVLE/ann_vle_model.onnx ANN_BinaryVLE/model_contract.json ./ANN_BinaryVLE/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
