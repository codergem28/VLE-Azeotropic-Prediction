"""Thermodynamic generation of the ethanol(1)-water(2) VLE dataset.

IMPORTANT: the data produced here is **not** experimental data. It is a
physics-based reference set generated from the NRTL activity-coefficient
model with Antoine vapour pressures. The ANN trained on it is a surrogate
for this thermodynamic model, and every accuracy metric in this project is
measured against that NRTL reference, not against laboratory measurements.

Conventions used throughout the project:
    x1, y1 : mole fraction of ethanol in liquid / vapour phase, [-]
    T      : temperature, K
    P      : pressure, bar

The NRTL parameters below are a literature fit for ethanol-water at
atmospheric pressure; the same parameters are used to build the reference
data and to compute the ideal (Raoult's law) baseline, so the comparison
between the two is self-consistent.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

# --------------------------------------------------------------------------
# Unit conversion
# --------------------------------------------------------------------------
# Antoine correlations below are in mmHg and °C; the project stores bar.
MMHG_PER_BAR = 750.061683

# --------------------------------------------------------------------------
# Antoine constants: log10(P_sat / mmHg) = A - B / (T / °C + C)
# --------------------------------------------------------------------------
ANTOINE = {
    'ethanol': {'A': 8.20417, 'B': 1642.89, 'C': 230.300},
    'water': {'A': 8.07131, 'B': 1730.63, 'C': 233.426},
}

# --------------------------------------------------------------------------
# NRTL parameters: tau_ij = a_ij + b_ij / T_K, alpha = 0.3
# --------------------------------------------------------------------------
NRTL = {
    'a12': -0.81, 'b12': 246.0,   # ethanol(1)-water(2)
    'a21': 3.46, 'b21': -586.0,
    'alpha': 0.3,
}

# Pressures of the generated isobars, in bar (200, 400, 760, 1000, 1500 mmHg).
PRESSURES_BAR = (
    200 / MMHG_PER_BAR,
    400 / MMHG_PER_BAR,
    760 / MMHG_PER_BAR,
    1000 / MMHG_PER_BAR,
    1500 / MMHG_PER_BAR,
)

# Isobar used for the "explicit T and P" VLE plot (~1 atm).
REFERENCE_PRESSURE_BAR = 760 / MMHG_PER_BAR

# Liquid composition grid: denser near the azeotrope (x1 ~ 0.8) and near
# x1 = 1, where y1 changes fastest. np.unique removes the duplicated grid
# points that would otherwise repeat rows across the train/test split.
X1_GRID = np.unique(np.round(np.concatenate([
    np.linspace(0.0, 0.7, 30),
    np.linspace(0.7, 0.95, 50),
    np.linspace(0.95, 1.0, 20),
]), 5))

FEATURES = ['x1', 'T', 'P']
TARGET = 'y1'

DATA_FILE = Path(__file__).resolve().parent / 'Data' / 'vle_data.csv'


def psat_mmHg(component, Tc):
    """Saturation pressure in mmHg from the Antoine correlation (Tc in °C)."""
    c = ANTOINE[component]
    return 10.0 ** (c['A'] - c['B'] / (Tc + c['C']))


def psat_bar(component, Tc):
    """Saturation pressure in bar from the Antoine correlation (Tc in °C)."""
    return psat_mmHg(component, Tc) / MMHG_PER_BAR


def psat_ethanol(Tc):
    """Ethanol saturation pressure in mmHg (Tc in °C)."""
    return psat_mmHg('ethanol', Tc)


def psat_water(Tc):
    """Water saturation pressure in mmHg (Tc in °C)."""
    return psat_mmHg('water', Tc)


def get_gammas(x1, Tk):
    """NRTL activity coefficients (gamma1, gamma2) at x1 and Tk [K]."""
    a12, b12 = NRTL['a12'], NRTL['b12']
    a21, b21 = NRTL['a21'], NRTL['b21']
    alpha = NRTL['alpha']

    x2 = 1.0 - x1
    tau12 = a12 + b12 / Tk
    tau21 = a21 + b21 / Tk
    G12 = np.exp(-alpha * tau12)
    G21 = np.exp(-alpha * tau21)

    gamma1 = np.exp(x2 ** 2 * (
        tau21 * (G21 / (x1 + x2 * G21)) ** 2
        + tau12 * G12 / (x2 + x1 * G12) ** 2
    ))
    gamma2 = np.exp(x1 ** 2 * (
        tau12 * (G12 / (x2 + x1 * G12)) ** 2
        + tau21 * G21 / (x1 + x2 * G21) ** 2
    ))
    return gamma1, gamma2


def pure_bubble_TK(component, P_bar):
    """Boiling temperature [K] of a pure component at P_bar.

    Solved directly from P_sat(T) = P. Activity coefficients are *not* used
    here: at x1 = 0 or x1 = 1 the liquid is a pure component, so its
    saturation condition is simply the pure-component one.
    """
    def resid(Tk):
        return psat_bar(component, Tk - 273.15) - P_bar

    # 250-500 K keeps the Antoine denominator positive and the bracket wide
    # enough for every pressure used here (0.27-2.0 bar).
    return brentq(resid, 250.0, 500.0, xtol=1e-10)


def bubble_TK(x1, P_bar):
    """Bubble temperature [K] at liquid composition x1 and pressure P_bar.

    Root of  x1*gamma1*Psat1 + x2*gamma2*Psat2 - P = 0  for a mixture, and the
    pure-component saturation condition at x1 = 0 and x1 = 1. Raises
    RuntimeError instead of falling back to a guessed temperature, so an
    unphysical row can never enter the dataset silently.
    """
    if not 0.0 <= x1 <= 1.0:
        raise ValueError(f'x1 must be a mole fraction in [0, 1], got {x1}')
    if P_bar <= 0:
        raise ValueError(f'P must be positive, got {P_bar} bar')

    # Pure-component limits: the activity-coefficient term is meaningless.
    if x1 == 0.0:
        return pure_bubble_TK('water', P_bar)
    if x1 == 1.0:
        return pure_bubble_TK('ethanol', P_bar)

    def resid(Tk):
        gamma1, gamma2 = get_gammas(x1, Tk)
        Tc = Tk - 273.15
        return (x1 * gamma1 * psat_bar('ethanol', Tc)
                + (1.0 - x1) * gamma2 * psat_bar('water', Tc) - P_bar)

    lo, hi = 250.0, 500.0
    f_lo, f_hi = resid(lo), resid(hi)
    if f_lo > 0 or f_hi < 0:
        raise RuntimeError(
            f'No bubble point bracket for x1={x1:.5f}, P={P_bar:.5f} bar '
            f'(residuals {f_lo:.4g} at {lo} K, {f_hi:.4g} at {hi} K)'
        )
    return brentq(resid, lo, hi, xtol=1e-10)


def vapor_composition(x1, Tk, P_bar):
    """Equilibrium vapour mole fraction y1 at (x1, Tk, P_bar).

    No clipping: the value is whatever modified Raoult's law gives, and
    validate_dataset() is what rejects unphysical rows.
    """
    if x1 == 0.0 or x1 == 1.0:
        return float(x1)  # pure component: y1 == x1 exactly

    gamma1, _ = get_gammas(x1, Tk)
    return float(x1 * gamma1 * psat_bar('ethanol', Tk - 273.15) / P_bar)


def generate_dataset(pressures_bar=PRESSURES_BAR, x1_grid=X1_GRID, decimals=5):
    """Build the NRTL reference VLE dataset.

    Returns a DataFrame with columns x1 [-], T [K], P [bar], y1 [-].
    """
    rows = []
    for P_bar in pressures_bar:
        for x1 in x1_grid:
            Tk = bubble_TK(float(x1), P_bar)
            y1 = vapor_composition(float(x1), Tk, P_bar)
            rows.append([x1, Tk, P_bar, y1])

    df = pd.DataFrame(rows, columns=FEATURES + [TARGET]).astype(float)
    return df.round(decimals)


def validate_dataset(df, pressure_tol=1e-4, composition_tol=2e-4):
    """Check that every row is a physically consistent NRTL equilibrium point.

    The generator is validated up front, so downstream code never has to
    "repair" bad values by clipping them.

    Returns:
        dict: summary statistics of the validated dataset.

    Raises:
        ValueError: if any physical or thermodynamic-consistency check fails.
    """
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f'Dataset is missing required column(s): {missing}')

    problems = []
    values = df[FEATURES + [TARGET]].values
    if not np.all(np.isfinite(values)):
        problems.append('dataset contains NaN or infinite values')
    if (df['x1'] < 0).any() or (df['x1'] > 1).any():
        problems.append('x1 outside [0, 1]')
    if (df['y1'] < 0).any() or (df['y1'] > 1).any():
        problems.append('y1 outside [0, 1]')
    if (df['P'] <= 0).any():
        problems.append('non-positive pressure')
    if (df['T'] <= 0).any():
        problems.append('non-positive temperature')
    if problems:
        raise ValueError('Dataset validation failed: ' + '; '.join(problems))

    # Ethanol-water has a single azeotrope: y1 - x1 is positive on the
    # ethanol-lean side, crosses zero once, and is negative beyond it. A
    # second sign change would mean the generator is inconsistent.
    for P_bar, group in df.groupby(df['P'].round(5)):
        interior = group[(group['x1'] > 0.0) & (group['x1'] < 1.0)].sort_values('x1')
        d = (interior['y1'] - interior['x1']).values
        signs = np.sign(np.where(np.abs(d) < composition_tol, 0.0, d))
        signs = signs[signs != 0]
        flips = int(np.sum(signs[1:] != signs[:-1]))
        if flips > 1:
            raise ValueError(
                f'P={P_bar} bar isobar has {flips} composition sign changes '
                '(y1 - x1), but ethanol-water admits only one azeotrope'
            )

    # Pure-component rows must sit on the pure saturation line at their P.
    for x1_val, component in ((0.0, 'water'), (1.0, 'ethanol')):
        pure = df[np.isclose(df['x1'], x1_val)]
        for _, row in pure.iterrows():
            expected_K = pure_bubble_TK(component, row['P'])
            if abs(row['T'] - expected_K) > 0.05:
                problems.append(
                    f'pure {component} at P={row["P"]:.5f} bar has '
                    f'T={row["T"]:.3f} K, expected {expected_K:.3f} K'
                )
            if abs(row['y1'] - x1_val) > composition_tol:
                problems.append(
                    f'pure {component} at P={row["P"]:.5f} bar has '
                    f'y1={row["y1"]:.5f}, expected {x1_val}'
                )

    # Every row must satisfy the bubble-pressure equation it claims to solve,
    # and reproduce its own y1 from that (x1, T, P).
    for _, row in df.iterrows():
        x1, Tk, P_bar = row['x1'], row['T'], row['P']
        if x1 == 0.0:
            calculated = psat_bar('water', Tk - 273.15)
        elif x1 == 1.0:
            calculated = psat_bar('ethanol', Tk - 273.15)
        else:
            g1, g2 = get_gammas(x1, Tk)
            calculated = (x1 * g1 * psat_bar('ethanol', Tk - 273.15)
                          + (1 - x1) * g2 * psat_bar('water', Tk - 273.15))

        if abs(calculated - P_bar) > pressure_tol:
            problems.append(
                f'row ({x1:.5f}, {Tk:.3f} K, {P_bar:.5f} bar) has bubble '
                f'pressure {calculated:.5f} bar'
            )

        expected_y1 = vapor_composition(x1, Tk, P_bar)
        if abs(expected_y1 - row['y1']) > composition_tol:
            problems.append(
                f'row ({x1:.5f}, {Tk:.3f} K, {P_bar:.5f} bar) has '
                f'y1={row["y1"]:.5f}, equilibrium gives {expected_y1:.5f}'
            )

    if problems:
        raise ValueError(
            'Dataset validation failed:\n  - ' + '\n  - '.join(problems[:10])
        )

    return {
        'n_rows': int(len(df)),
        'pressures_bar': sorted(df['P'].round(5).unique().tolist()),
        'x1_range': [float(df['x1'].min()), float(df['x1'].max())],
        'T_range_K': [float(df['T'].min()), float(df['T'].max())],
        'y1_range': [float(df['y1'].min()), float(df['y1'].max())],
    }


def main():
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    df = generate_dataset()
    summary = validate_dataset(df)
    df.to_csv(DATA_FILE, index=False)

    print(f'Wrote {summary["n_rows"]} NRTL equilibrium points to {DATA_FILE}')
    print(f'  pressures [bar]: {[round(p, 5) for p in summary["pressures_bar"]]}')
    print(f'  x1 range [-]:    {summary["x1_range"]}')
    print(f'  T range [K]:     {[round(v, 2) for v in summary["T_range_K"]]}')
    print(f'  y1 range [-]:    {[round(v, 5) for v in summary["y1_range"]]}')


if __name__ == '__main__':
    main()
