"""HTTP inference service for the ethanol-water VLE ANN surrogate.

Run locally:
    uvicorn serve:app --reload --port 8000

Contract: every prediction takes rows of [x1, T(K), P(bar)] and returns y1 in
[-]. The model is ONNX Runtime executing the trained Keras network; scaling is
reproduced from `model_contract.json`, so no TensorFlow, scikit-learn or pandas
is needed at runtime. The model is a surrogate for the NRTL reference data in
data.py, not for experimental measurements -- /metadata reports that next to the
scores so a caller cannot mistake one for the other.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'ANN_BinaryVLE'
MODEL_PATH = OUTPUT_DIR / 'ann_vle_model.onnx'
CONTRACT_PATH = OUTPUT_DIR / 'model_contract.json'

STATE = {}


@asynccontextmanager
async def lifespan(app):
    for path in (MODEL_PATH, CONTRACT_PATH):
        if not path.exists():
            raise RuntimeError(f'{path} not found: run export_onnx.py first')

    contract = json.loads(CONTRACT_PATH.read_text(encoding='utf-8'))
    STATE['contract'] = contract
    STATE['session'] = ort.InferenceSession(str(MODEL_PATH),
                                           providers=['CPUExecutionProvider'])
    STATE['input_name'] = STATE['session'].get_inputs()[0].name
    STATE['mean'] = np.asarray(contract['scaler']['mean'], dtype=np.float64)
    STATE['scale'] = np.asarray(contract['scaler']['scale'], dtype=np.float64)
    yield
    STATE.clear()


app = FastAPI(
    title='Ethanol-water VLE ANN surrogate',
    description=(
        'Predicts vapour-phase ethanol mole fraction y1 from liquid composition x1, '
        'temperature T [K] and pressure P [bar]. Trained on NRTL reference data, '
        'not on experimental measurements.'
    ),
    version='1.2.0',
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
    contract = STATE['contract']
    feature_names = contract['feature_names']
    messages = []

    if rows.ndim != 2 or rows.shape[1] != len(feature_names):
        raise HTTPException(
            status_code=422,
            detail=f'expected N rows of {len(feature_names)} values {feature_names}, '
                   f'got shape {rows.shape}',
        )
    if not np.all(np.isfinite(rows)):
        raise HTTPException(status_code=422, detail='inputs must be finite numbers')
    if (rows[:, 0] < 0).any() or (rows[:, 0] > 1).any():
        raise HTTPException(status_code=422, detail='x1 must be a mole fraction in [0, 1]')
    if (rows[:, 1] <= 0).any() or (rows[:, 2] <= 0).any():
        raise HTTPException(status_code=422, detail='T [K] and P [bar] must be positive')

    for name, (lo, hi) in contract['feature_ranges'].items():
        i = feature_names.index(name)
        outside = (rows[:, i] < lo) | (rows[:, i] > hi)
        if outside.any():
            messages.append(
                f'{int(outside.sum())} input(s) have {name} outside the trained range '
                f'[{lo:.5f}, {hi:.5f}]: prediction is extrapolation'
            )

    scaled = ((rows - STATE['mean']) / STATE['scale']).astype(np.float32)
    y = STATE['session'].run(None, {STATE['input_name']: scaled})[0].flatten()
    return np.clip(y, 0.0, 1.0), messages


@app.get('/', include_in_schema=False)
def root():
    """Send the bare hostname somewhere useful instead of a 404."""
    return RedirectResponse(url='/docs')


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model': MODEL_PATH.name,
        'runtime': f'onnxruntime {ort.__version__}',
        'reference_data': STATE['contract'].get('reference_data'),
    }


@app.get('/metadata')
def metadata():
    contract = STATE['contract']
    return {
        'feature_names': contract['feature_names'],
        'target_name': contract['target_name'],
        'units': contract['units'],
        'feature_ranges': contract['feature_ranges'],
        'reference_data': contract['reference_data'],
        'metrics': contract.get('metrics'),
        'export_verification': contract.get('verification'),
    }


@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    y, messages = _predict(np.asarray(request.inputs, dtype=np.float64))
    return PredictResponse(predictions=[float(v) for v in y], warnings=messages)
