# predict_from_saved_model.py
"""Load the trained ANN surrogate and predict ethanol vapour composition.

Conventions: x1, y1 [-] mole fractions, T [K], P [bar]. The model was trained
on the NRTL + Antoine reference data from data.py, so predictions are
surrogates for that thermodynamic model, not for experimental data.
"""

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras

from data import (
    FEATURES,
    REFERENCE_PRESSURE_BAR,
    TARGET,
    bubble_TK,
    vapor_composition,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'ANN_BinaryVLE'
DEFAULT_MODEL_PATH = OUTPUT_DIR / 'ann_vle_model.h5'
DEFAULT_PIPELINE_PATH = OUTPUT_DIR / 'preprocessing_pipeline.pkl'
DATA_FILE = BASE_DIR / 'Data' / 'vle_data.csv'

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def _check_domain(X_raw, pipeline):
    """Warn when a prediction request falls outside the training domain."""
    ranges = pipeline.get('feature_ranges')
    if not ranges:
        return
    for i, name in enumerate(pipeline.get('feature_names', FEATURES)):
        lo, hi = ranges[name]
        out = (X_raw[:, i] < lo - 1e-9) | (X_raw[:, i] > hi + 1e-9)
        if out.any():
            warnings.warn(
                f'{int(out.sum())} input(s) have {name} outside the trained '
                f'range [{lo:.5f}, {hi:.5f}]: predictions are extrapolation.',
                RuntimeWarning,
                stacklevel=2,
            )
    if (X_raw[:, 0] < 0).any() or (X_raw[:, 0] > 1).any():
        raise ValueError('x1 must be a mole fraction in [0, 1]')


def predict_from_saved(X_raw, model_path=DEFAULT_MODEL_PATH,
                       pipeline_path=DEFAULT_PIPELINE_PATH):
    """Predict y1 for raw rows of (x1 [mol/mol], T [K], P [bar])."""
    model = keras.models.load_model(str(model_path), compile=False)
    preprocessing_pipeline = joblib.load(pipeline_path)
    scaler = preprocessing_pipeline['scaler']

    X_raw = np.asarray(X_raw, dtype=float)
    if X_raw.ndim == 1:
        X_raw = X_raw.reshape(1, -1)
    if X_raw.shape[1] != len(FEATURES):
        raise ValueError(
            f'Expected {len(FEATURES)} columns {FEATURES}, got shape {X_raw.shape}'
        )

    _check_domain(X_raw, preprocessing_pipeline)

    X_scaled = scaler.transform(X_raw)
    y_pred = model.predict(X_scaled, verbose=0).flatten()
    return np.clip(y_pred, 0, 1)


def recreate_plots(model_path=DEFAULT_MODEL_PATH,
                   pipeline_path=DEFAULT_PIPELINE_PATH,
                   data_file=DATA_FILE,
                   P_bar=REFERENCE_PRESSURE_BAR):
    """Recreate the evaluation plots from the saved artifacts."""
    df = pd.read_csv(data_file)
    X = df[FEATURES].values
    y = df[TARGET].values

    model = keras.models.load_model(str(model_path), compile=False)
    preprocessing_pipeline = joblib.load(pipeline_path)
    scaler = preprocessing_pipeline['scaler']
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled, verbose=0).flatten()
    y_pred = np.clip(y_pred, 0, 1)

    # Parity plot
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y, y=y_pred, alpha=0.7, s=60)
    min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
             label='Perfect prediction')
    plt.xlabel('NRTL reference y1 [-]')
    plt.ylabel('ANN predicted y1 [-]')
    plt.title('Parity Plot: NRTL Reference vs ANN Prediction')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'parity_plot_recreated.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Residuals plot
    residuals = y_pred - y
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y, y=residuals, alpha=0.7, s=60)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero residual')
    plt.xlabel('NRTL reference y1 [-]')
    plt.ylabel('Residuals: ANN - NRTL [-]')
    plt.title(
        f'Residuals Plot (MAE = {mean_absolute_error(y, y_pred):.6f}, '
        f'RMSE = {np.sqrt(mean_squared_error(y, y_pred)):.6f})'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'residuals_plot_recreated.png', dpi=300, bbox_inches='tight')
    plt.close()

    # y1 vs x1 for one explicit pressure; T follows the NRTL bubble curve.
    x1_grid = np.linspace(0.0, 1.0, 201)
    T_bubble = np.array([bubble_TK(float(x), P_bar) for x in x1_grid])
    y1_nrtl = np.array([vapor_composition(float(x), float(t), P_bar)
                        for x, t in zip(x1_grid, T_bubble)])

    X_grid = np.column_stack([x1_grid, T_bubble, np.full_like(x1_grid, P_bar)])
    y_grid = np.clip(model.predict(scaler.transform(X_grid), verbose=0).flatten(), 0, 1)

    on_isobar = np.isclose(X[:, 2], P_bar, atol=1e-4)

    plt.figure(figsize=(10, 6))
    plt.plot(x1_grid, y1_nrtl, 'g-', linewidth=2.5, label='NRTL reference (bubble curve)')
    plt.plot(x1_grid, y_grid, 'b--', linewidth=2, label='ANN surrogate')
    plt.plot(x1_grid, x1_grid, 'r:', linewidth=2, label='y1 = x1 (azeotrope line)')
    plt.scatter(X[on_isobar, 0], y[on_isobar], color='black', s=45, alpha=0.8,
                label='NRTL data on this isobar', zorder=5)
    plt.xlabel('x1 [-], ethanol liquid mole fraction')
    plt.ylabel('y1 [-], ethanol vapour mole fraction')
    plt.title(
        f'VLE isobar at P = {P_bar:.4f} bar (760 mmHg, ~1 atm)\n'
        f'T = NRTL bubble temperature, {T_bubble.min():.1f}-{T_bubble.max():.1f} K'
    )
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'y_vs_x_plot_recreated.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    if not DEFAULT_MODEL_PATH.exists() or not DEFAULT_PIPELINE_PATH.exists():
        raise FileNotFoundError(
            "Model and/or pipeline not found. Run preprocessing_training_saving.py first."
        )
    recreate_plots()
