# export_onnx.py
"""Export the trained Keras surrogate to ONNX for the inference container.

Why: the Keras model is 240 KB, but serving it pulls TensorFlow (~3.4 GB image).
ONNX Runtime runs the same network in ~50 MB, so the API image stays small and
cold starts on free hosts are seconds instead of a minute.

The scaler is exported as plain mean/scale arrays into `model_contract.json`, so
the service needs neither scikit-learn nor scipy to reproduce training-time
scaling. The contract is verified against the Keras model on every row of the
training data before it is written.

Usage:
    python export_onnx.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import tensorflow as tf
import tf2onnx
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'ANN_BinaryVLE'
MODEL_PATH = OUTPUT_DIR / 'ann_vle_model.h5'
PIPELINE_PATH = OUTPUT_DIR / 'preprocessing_pipeline.pkl'
ONNX_PATH = OUTPUT_DIR / 'ann_vle_model.onnx'
CONTRACT_PATH = OUTPUT_DIR / 'model_contract.json'
DATA_FILE = BASE_DIR / 'Data' / 'vle_data.csv'
METRICS_FILE = OUTPUT_DIR / 'metrics.json'

TOLERANCE = 1e-5


def export_onnx(model, onnx_path=ONNX_PATH):
    """Convert the Keras 3 model through a traced tf.function.

    tf2onnx's from_keras helper mis-handles Keras 3 output naming, so the model
    is wrapped in a tf.function with an explicit signature instead.
    """
    spec = [tf.TensorSpec([None, 3], tf.float32, name='features')]

    @tf.function(input_signature=spec)
    def _forward(features):
        return {'y1': model(features, training=False)}

    tf2onnx.convert.from_function(_forward, input_signature=spec, opset=17,
                                  output_path=str(onnx_path))
    return onnx_path


def build_contract(pipeline):
    scaler = pipeline['scaler']
    contract = {
        'feature_names': list(pipeline['feature_names']),
        'target_name': pipeline['target_name'],
        'units': pipeline['units'],
        'feature_ranges': {k: [float(v[0]), float(v[1])]
                           for k, v in pipeline['feature_ranges'].items()},
        'scaler': {
            'type': type(scaler).__name__,
            'mean': [float(v) for v in scaler.mean_],
            'scale': [float(v) for v in scaler.scale_],
        },
        'reference_data': pipeline.get('reference_data'),
        'model_file': ONNX_PATH.name,
    }
    if METRICS_FILE.exists():
        contract['metrics'] = json.loads(METRICS_FILE.read_text(encoding='utf-8'))
    return contract


def main():
    for path in (MODEL_PATH, PIPELINE_PATH):
        if not path.exists():
            raise FileNotFoundError(f'{path} not found: run preprocessing_training_saving.py first')

    model = keras.models.load_model(str(MODEL_PATH), compile=False)
    pipeline = joblib.load(PIPELINE_PATH)

    export_onnx(model)
    print(f'wrote {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1024:.1f} KB)')

    contract = build_contract(pipeline)
    contract['model_file'] = ONNX_PATH.name

    # Verify ONNX against Keras on the whole dataset through the exported scaling.
    df = pd.read_csv(DATA_FILE)
    X_raw = df[contract['feature_names']].to_numpy(dtype=np.float64)
    mean = np.asarray(contract['scaler']['mean'])
    scale = np.asarray(contract['scaler']['scale'])
    X_scaled = ((X_raw - mean) / scale).astype(np.float32)

    y_keras = np.clip(model.predict(X_scaled, verbose=0).flatten(), 0.0, 1.0)

    session = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    y_onnx = np.clip(session.run(None, {input_name: X_scaled})[0].flatten(), 0.0, 1.0)

    diff = np.abs(y_keras - y_onnx)
    print(f'rows={len(y_onnx)}  max|keras-onnx|={diff.max():.3e}  mean={diff.mean():.3e}')
    if diff.max() > TOLERANCE:
        raise SystemExit(f'ONNX export diverged from Keras: max diff {diff.max():.3e}')

    contract['verification'] = {
        'rows': int(len(y_onnx)),
        'max_abs_diff_vs_keras': float(diff.max()),
        'tolerance': TOLERANCE,
    }
    CONTRACT_PATH.write_text(json.dumps(contract, indent=2), encoding='utf-8')
    print(f'wrote {CONTRACT_PATH}')
    print('sample:', {k: round(float(v), 5) for k, v in
                      zip(contract['feature_names'], X_raw[0])}, '-> y1', round(float(y_onnx[0]), 5))


if __name__ == '__main__':
    main()
