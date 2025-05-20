import logging
from typing import List, Optional
import numpy as np
from scipy import stats
from pathlib import Path
import json

import pandas as pd
import statsmodels.api as sm
from pingouin import partial_corr
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def merge_and_resample(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Join dataframes on a daily frequency using the mean."""
    df = None
    for d in dfs:
        d = d.set_index("timestamp")
        df = d if df is None else df.join(d, how="inner")
    df = df.resample("D").mean().dropna().reset_index()
    return df


def run_three_way_regression(
    df: pd.DataFrame,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS model of movement against rainfall and temperature."""
    X = df[["rainfall_mm", "temperature_C"]]
    X = sm.add_constant(X)
    y = df["movement_mm"]
    model = sm.OLS(y, X).fit()
    logger.info("\n%s", model.summary())
    ci90 = model.conf_int(alpha=0.10)
    logger.info("90%% confidence intervals:\n%s", ci90)
    return model


def compute_partial_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute partial correlations using pingouin."""
    p1 = partial_corr(data=df, x="rainfall_mm", y="movement_mm", covar="temperature_C")
    p2 = partial_corr(data=df, x="temperature_C", y="movement_mm", covar="rainfall_mm")
    return pd.concat([p1, p2], axis=0)


def compute_rolling_correlation(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Return rolling correlation and p-values between movement and rainfall."""
    records = []
    for i in range(window - 1, len(df)):
        sub = df.iloc[i - window + 1 : i + 1]
        r, p = stats.pearsonr(sub["movement_mm"], sub["rainfall_mm"])
        records.append(
            {
                "timestamp": df["timestamp"].iloc[i],
                "r": float(r),
                "p": float(p),
            }
        )
    return pd.DataFrame(records)


def first_significant_date(
    rcorr: pd.DataFrame,
    r_thresh: float = 0.4,
    p_thresh: float = 0.05,
    consecutive: int = 1,
) -> Optional[pd.Timestamp]:
    """Find the first date where correlation exceeds thresholds."""
    mask = (rcorr["r"].abs() >= r_thresh) & (rcorr["p"] < p_thresh)
    if consecutive > 1:
        rolled = mask.rolling(consecutive).apply(lambda x: all(x), raw=True)
        idx = rolled.idxmax() if (rolled >= 1).any() else None
    else:
        idx = mask.idxmax() if mask.any() else None
    if idx is not None and mask.iloc[idx]:
        return rcorr.loc[idx, "timestamp"]
    return None


def compute_ccf(x, y, max_lag: int = 30) -> List[float]:
    """Compute cross-correlation up to ``max_lag`` positive lags."""
    try:
        from statsmodels.tsa.stattools import ccf as sm_ccf

        return sm_ccf(x, y)[:max_lag].tolist()
    except Exception:  # pragma: no cover - fallback implementation
        x = np.asarray(x) - np.mean(x)
        y = np.asarray(y) - np.mean(y)
        n = len(x)
        corr = np.correlate(x, y, mode="full")
        mid = n - 1
        denom = np.std(x) * np.std(y) * n
        return (corr[mid : mid + max_lag] / denom).tolist()


def compute_tvccf_peaks(
    df: pd.DataFrame,
    window: int = 60,
    max_lag: int = 30,
    thresh: float = 0.4,
    output_dir: Path = Path("."),
) -> Optional[pd.Timestamp]:
    """Track peak cross-correlation lag and value over time."""
    records = []
    for i in range(window, len(df)):
        sub = df.iloc[i - window : i]
        vals = compute_ccf(sub["rainfall_mm"], sub["movement_mm"], max_lag=max_lag)
        idx = int(np.argmax(np.abs(vals)))
        records.append(
            {
                "timestamp": df["timestamp"].iloc[i],
                "lag": int(idx),
                "r": float(vals[idx]),
            }
        )
    res = pd.DataFrame(records)
    onset = None
    if not res.empty:
        res.to_json(
            output_dir / "tvccf_peaks.json", orient="records", date_format="iso"
        )
        plt.plot(res["timestamp"], res["lag"], label="Lag")
        plt.twinx()
        plt.plot(res["timestamp"], res["r"], color="orange", label="Correlation")
        plt.axhline(thresh, color="red", linestyle="--")
        plt.axhline(-thresh, color="red", linestyle="--")
        plt.title("Time-Varying CCF Peaks")
        plt.legend(loc="upper left")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "tvccf_peaks.png")
        plt.clf()
        mask = res["r"].abs() >= thresh
        if mask.any():
            onset = res.loc[mask.idxmax(), "timestamp"]
    return onset


def compute_sliding_granger(
    df: pd.DataFrame,
    window: int = 60,
    maxlag: int = 1,
    alpha: float = 0.05,
    output_dir: Path = Path("."),
) -> Optional[pd.Timestamp]:
    """Sliding-window Granger causality test of rainfall -> movement."""
    from statsmodels.tsa.stattools import grangercausalitytests

    records = []
    for i in range(window, len(df)):
        sub = df[["movement_mm", "rainfall_mm"]].iloc[i - window : i]
        try:
            test = grangercausalitytests(sub, maxlag=maxlag, verbose=False)
            pval = test[maxlag][0]["ssr_ftest"][1]
        except Exception:
            pval = np.nan
        records.append({"timestamp": df["timestamp"].iloc[i], "p": float(pval)})
    res = pd.DataFrame(records)
    onset = None
    if not res.empty:
        res.to_json(
            output_dir / "granger_causality.json", orient="records", date_format="iso"
        )
        plt.plot(res["timestamp"], res["p"])
        plt.axhline(alpha, color="red", linestyle="--")
        plt.ylabel("p-value")
        plt.title("Sliding-Window Granger Causality")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "granger_causality.png")
        plt.clf()
        mask = res["p"] < alpha
        if mask.any():
            onset = res.loc[mask.idxmax(), "timestamp"]
    return onset


def compute_change_point_correlation(
    df: pd.DataFrame, corr_window: int = 7, output_dir: Path = Path(".")
) -> Optional[pd.Timestamp]:
    """Detect change points in a correlation time series."""
    records = []
    for i in range(corr_window, len(df)):
        sub = df.iloc[i - corr_window : i]
        r, _ = stats.pearsonr(sub["movement_mm"], sub["rainfall_mm"])
        records.append({"timestamp": df["timestamp"].iloc[i], "r": float(r)})
    series = np.array([r["r"] for r in records])
    cp_idx = None
    onset = None
    try:
        import ruptures as rpt

        algo = rpt.Pelt(model="rbf").fit(series)
        result = algo.predict(pen=3)
        cp_idx = result[0] - 1 if result else None
        if cp_idx is not None and cp_idx < len(records):
            onset = records[cp_idx]["timestamp"]
    except Exception:
        cp_idx = None
    corr_df = pd.DataFrame(records)
    if not corr_df.empty:
        corr_df.to_json(
            output_dir / "correlation_change_points.json",
            orient="records",
            date_format="iso",
        )
        plt.plot(corr_df["timestamp"], corr_df["r"], label="Correlation")
        if cp_idx is not None:
            plt.axvline(
                corr_df["timestamp"][cp_idx],
                color="red",
                linestyle="--",
                label="Change",
            )
        plt.legend()
        plt.title("Correlation Change-Point Detection")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_change_points.png")
        plt.clf()
    return onset


def compute_wavelet_coherence(
    df: pd.DataFrame, output_dir: Path = Path(".")
) -> Optional[pd.Timestamp]:
    """Compute wavelet coherence between rainfall and movement."""
    try:
        import pywt
    except Exception:
        logger.warning("pywt not installed, skipping wavelet coherence analysis")
        return None
    scales = np.arange(1, 64)
    cwt_x, _ = pywt.cwt(df["rainfall_mm"], scales, "morl")
    cwt_y, _ = pywt.cwt(df["movement_mm"], scales, "morl")
    Wxy = cwt_x * np.conj(cwt_y)
    Sxx = np.abs(cwt_x) ** 2
    Syy = np.abs(cwt_y) ** 2
    coherence = np.abs(Wxy) ** 2 / (Sxx * Syy)
    power = coherence.mean(axis=0)
    onset_idx = np.argmax(power > 0.5) if np.any(power > 0.5) else None
    onset = df["timestamp"].iloc[onset_idx] if onset_idx is not None else None
    plt.imshow(
        coherence,
        aspect="auto",
        origin="lower",
        extent=[0, len(df), scales[0], scales[-1]],
    )
    plt.xlabel("Time index")
    plt.ylabel("Scale")
    plt.title("Wavelet Coherence")
    plt.colorbar(label="Coherence")
    plt.tight_layout()
    plt.savefig(output_dir / "wavelet_coherence.png")
    plt.clf()
    return onset


def compute_tvp_regression(
    df: pd.DataFrame,
    window: int = 60,
    threshold: float = 0.0,
    output_dir: Path = Path("."),
) -> Optional[pd.Timestamp]:
    """Estimate time-varying rainfall coefficient via rolling OLS."""
    records = []
    for i in range(window, len(df)):
        sub = df.iloc[i - window : i]
        X = sm.add_constant(sub["rainfall_mm"])
        y = sub["movement_mm"]
        res = sm.OLS(y, X).fit()
        coef = res.params["rainfall_mm"]
        records.append({"timestamp": df["timestamp"].iloc[i], "coef": float(coef)})
    df_coef = pd.DataFrame(records)
    onset = None
    if not df_coef.empty:
        df_coef.to_json(
            output_dir / "tvp_regression.json", orient="records", date_format="iso"
        )
        plt.plot(df_coef["timestamp"], df_coef["coef"])
        plt.axhline(threshold, color="red", linestyle="--")
        plt.ylabel("Coefficient")
        plt.title("Time-Varying Parameter Regression")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "tvp_regression.png")
        plt.clf()
        mask = df_coef["coef"] > threshold
        if mask.any():
            onset = df_coef.loc[mask.idxmax(), "timestamp"]
    return onset


def _mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Estimate mutual information using a 2D histogram."""
    c_xy = np.histogram2d(x, y, bins)[0]
    p_xy = c_xy / c_xy.sum()
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    nz = p_xy > 0
    return float(
        (p_xy[nz] * np.log(p_xy[nz] / (p_x[:, None] * p_y[None, :])[nz])).sum()
    )


def compute_mutual_information(
    df: pd.DataFrame,
    window: int = 60,
    bins: int = 10,
    threshold: float = 0.1,
    output_dir: Path = Path("."),
) -> Optional[pd.Timestamp]:
    """Sliding-window mutual information between rainfall and movement."""
    records = []
    for i in range(window, len(df)):
        sub = df.iloc[i - window : i]
        mi = _mutual_information(
            sub["rainfall_mm"].values, sub["movement_mm"].values, bins=bins
        )
        records.append({"timestamp": df["timestamp"].iloc[i], "mi": float(mi)})
    mi_df = pd.DataFrame(records)
    onset = None
    if not mi_df.empty:
        mi_df.to_json(
            output_dir / "mutual_information.json", orient="records", date_format="iso"
        )
        plt.plot(mi_df["timestamp"], mi_df["mi"])
        plt.axhline(threshold, color="red", linestyle="--")
        plt.ylabel("Mutual Information")
        plt.title("Mutual Information Window")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "mutual_information.png")
        plt.clf()
        mask = mi_df["mi"] >= threshold
        if mask.any():
            onset = mi_df.loc[mask.idxmax(), "timestamp"]
    return onset


def compute_hmm_regime(
    df: pd.DataFrame, n_states: int = 2, output_dir: Path = Path(".")
) -> Optional[pd.Timestamp]:
    """Detect coupling regimes using a hidden Markov model."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception:
        logger.warning("hmmlearn not installed, skipping HMM analysis")
        return None
    X = df[["rainfall_mm", "movement_mm"]].values
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)
    model.fit(X)
    states = model.predict(X)
    means = [df["movement_mm"][states == i].mean() for i in range(n_states)]
    high_state = int(np.argmax(means))
    change_idx = int(np.argmax(states == high_state))
    onset = df["timestamp"].iloc[change_idx]
    plt.plot(df["timestamp"], states, drawstyle="steps-post")
    plt.ylabel("State")
    plt.title("HMM Regime Detection")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "hmm_states.png")
    plt.clf()
    pd.DataFrame({"timestamp": df["timestamp"], "state": states}).to_json(
        output_dir / "hmm_states.json", orient="records", date_format="iso"
    )
    return onset


def analyze_movement_rain_temp(
    df: pd.DataFrame, output_dir: Path = Path("analysis_outputs")
) -> None:
    """Run three-way analysis of movement, rainfall and temperature.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe containing ``timestamp``, ``movement_mm``, ``rainfall_mm``
        and ``temperature_C`` columns already aligned on a daily frequency.
    output_dir : Path, optional
        Directory where result plots will be written. Created if it does not
        already exist.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.debug("Beginning correlation analysis with %d records", len(df))

    # Save merged data for interactive plotting
    (output_dir / "merged_data.json").write_text(
        df.to_json(orient="split", date_format="iso")
    )

    logger.info("Running regression analysis...")
    model = run_three_way_regression(df)

    logger.info("Computing partial correlations...")
    pcorrs = compute_partial_correlations(df)
    logger.info("Partial correlations:\n%s", pcorrs)

    logger.info("Computing rolling correlation...")
    rcorr = compute_rolling_correlation(df, window=10)
    first_date = first_significant_date(rcorr, r_thresh=0.4)
    if not rcorr.empty:
        plt.plot(rcorr["timestamp"], rcorr["r"])
        plt.axhline(0, color="black", linewidth=0.5)
        plt.xlabel("Date")
        plt.ylabel("Rolling Pearson r")
        plt.title("Rolling 10-day correlation")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "rolling_correlation.png")
        plt.clf()
    else:
        first_date = None

    logger.info("Generating plots...")
    pd.plotting.scatter_matrix(
        df[["movement_mm", "rainfall_mm", "temperature_C"]], diagonal="kde"
    )
    plt.suptitle("Scatter Matrix")
    plt.savefig(output_dir / "scatter_matrix.png")
    plt.clf()

    rainfall_ccf = compute_ccf(df["rainfall_mm"], df["movement_mm"])
    temperature_ccf = compute_ccf(df["temperature_C"], df["movement_mm"])

    (output_dir / "rainfall_ccf.json").write_text(json.dumps(rainfall_ccf))
    (output_dir / "temperature_ccf.json").write_text(json.dumps(temperature_ccf))

    def plot_ccf(values, name):
        plt.bar(range(len(values)), values)
        plt.title(f"{name} → Movement CCF")
        plt.xlabel("Lag (days)")
        plt.ylabel("Correlation")
        plt.savefig(output_dir / f"{name.lower()}_ccf.png")
        plt.clf()

    plot_ccf(rainfall_ccf, name="Rainfall")
    plot_ccf(temperature_ccf, name="Temperature")

    logger.info("Computing time-varying CCF peaks...")
    tvccf_date = compute_tvccf_peaks(df, output_dir=output_dir)

    logger.info("Computing sliding-window Granger causality...")
    granger_date = compute_sliding_granger(df, output_dir=output_dir)

    logger.info("Detecting change points in correlation series...")
    cp_date = compute_change_point_correlation(df, output_dir=output_dir)

    logger.info("Running wavelet coherence analysis...")
    wavelet_date = compute_wavelet_coherence(df, output_dir=output_dir)

    logger.info("Fitting time-varying parameter regression...")
    tvp_date = compute_tvp_regression(df, output_dir=output_dir)

    logger.info("Computing mutual information window...")
    mi_date = compute_mutual_information(df, output_dir=output_dir)

    logger.info("Detecting regimes via HMM...")
    hmm_date = compute_hmm_regime(df, output_dir=output_dir)

    summary = {
        "regression": {
            "coefficients": model.params.to_dict(),
            "p_values": model.pvalues.to_dict(),
            "r_squared": model.rsquared,
        },
        "partial_correlations": {
            "rainfall_vs_movement_given_temp": pcorrs.iloc[0].to_dict(),
            "temperature_vs_movement_given_rain": pcorrs.iloc[1].to_dict(),
        },
        "rolling_correlation": {
            "window_days": 10,
            "threshold_r": 0.4,
            "threshold_p": 0.05,
            "first_significant_date": (
                first_date.strftime("%Y-%m-%d") if first_date is not None else None
            ),
        },
        "tvccf_peak": tvccf_date.strftime("%Y-%m-%d") if tvccf_date else None,
        "granger_causality": (
            granger_date.strftime("%Y-%m-%d") if granger_date else None
        ),
        "correlation_change_point": cp_date.strftime("%Y-%m-%d") if cp_date else None,
        "wavelet_coherence": (
            wavelet_date.strftime("%Y-%m-%d") if wavelet_date else None
        ),
        "tvp_regression": tvp_date.strftime("%Y-%m-%d") if tvp_date else None,
        "mutual_information": mi_date.strftime("%Y-%m-%d") if mi_date else None,
        "hmm_regime": hmm_date.strftime("%Y-%m-%d") if hmm_date else None,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    logger.info("Analysis complete. Plots saved to %s", output_dir.resolve())
