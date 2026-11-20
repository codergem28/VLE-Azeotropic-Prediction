"""HTTP inference service for the ethanol-water VLE ANN surrogate.

Run locally:
    uvicorn serve:app --reload --port 8000

Contract: every prediction takes rows of [x1, T(K), P(bar)] and returns y1 in
[-]. The model is a surrogate for the NRTL reference data in data.py, not for
experimental measurements -- /metadata reports that next to the scores so a
caller cannot mistake one for the other.
"""

import json
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow import keras

from data import FEATURES

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'ANN_BinaryVLE'
MODEL_PATH = OUTPUT_DIR / 'ann_vle_model.h5'
PIPELINE_PATH = OUTPUT_DIR / 'preprocessing_pipeline.pkl'
METRICS_PATH = OUTPUT_DIR / 'metrics.json'

STATE = {}


@asynccontextmanager
async def lifespan(app):
    for path in (MODEL_PATH, PIPELINE_PATH):
        if not path.exists():
            raise RuntimeError(f'{path} not found: run preprocessing_training_saving.py first')
    STATE['model'] = keras.models.load_model(str(MODEL_PATH), compile=False)
    STATE['pipeline'] = joblib.load(PIPELINE_PATH)
    yield
    STATE.clear()


app = FastAPI(
    title='Ethanol-water VLE ANN surrogate',
    description=(
        'Predicts vapour-phase ethanol mole fraction y1 from liquid composition x1, '
        'temperature T [K] and pressure P [bar]. Trained on NRTL reference data, '
        'not on experimental measurements.'
    ),
    version='1.0.0',
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    inputs: list[tuple[float, float, float]] = Field(
        ...,
        description='Rows of [x1 (mol/mol), T (K), P (bar)]',
        min_length=1,
    )

    model_config = {
        'json_schema_extra': {
            'examples': [{'inputs': [[0.5, 350.0, 1.01325], [0.9, 351.3, 1.01325]]}]
        }
    }


class PredictResponse(BaseModel):
    predictions: list[float] = Field(..., description='y1 (mol/mol) for each input row')
    warnings: list[str] = Field(default_factory=list)


def _predict(rows: np.ndarray) -> tuple[np.ndarray, list[str]]:
    pipeline = STATE['pipeline']
    messages = []

    if rows.ndim != 2 or rows.shape[1] != len(FEATURES):
        raise HTTPException(
            status_code=422,
            detail=f'expected N rows of {len(FEATURES)} values {FEATURES}, got shape {rows.shape}',
        )
    if not np.all(np.isfinite(rows)):
        raise HTTPException(status_code=422, detail='inputs must be finite numbers')
    if (rows[:, 0] < 0).any() or (rows[:, 0] > 1).any():
        raise HTTPException(status_code=422, detail='x1 must be a mole fraction in [0, 1]')
    if (rows[:, 1] <= 0).any() or (rows[:, 2] <= 0).any():
        raise HTTPException(status_code=422, detail='T [K] and P [bar] must be positive')

    for name, (lo, hi) in pipeline.get('feature_ranges', {}).items():
        i = pipeline['feature_names'].index(name)
        outside = (rows[:, i] < lo) | (rows[:, i] > hi)
        if outside.any():
            messages.append(
                f'{int(outside.sum())} input(s) have {name} outside the trained range '
                f'[{lo:.5f}, {hi:.5f}]: prediction is extrapolation'
            )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        y = STATE['model'].predict(pipeline['scaler'].transform(rows), verbose=0).flatten()
    messages.extend(str(w.message) for w in caught)

    return np.clip(y, 0.0, 1.0), messages


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model': MODEL_PATH.name,
        'reference_data': STATE['pipeline'].get('reference_data'),
    }


@app.get('/metadata')
def metadata():
    body = {
        'feature_names': STATE['pipeline'].get('feature_names'),
        'target_name': STATE['pipeline'].get('target_name'),
        'units': STATE['pipeline'].get('units'),
        'feature_ranges': STATE['pipeline'].get('feature_ranges'),
        'reference_data': STATE['pipeline'].get('reference_data'),
    }
    if METRICS_PATH.exists():
        body['metrics'] = json.loads(METRICS_PATH.read_text(encoding='utf-8'))
    return body


@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    y, messages = _predict(np.asarray(request.inputs, dtype=float))
    return PredictResponse(predictions=[float(v) for v in y], warnings=messages)
