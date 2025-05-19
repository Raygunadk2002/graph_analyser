import logging
from typing import List
from pathlib import Path
import json

import pandas as pd
import statsmodels.api as sm
from pingouin import partial_corr
import matplotlib.pyplot as plt

from data_loader import load_monitoring_data, load_rainfall_data, load_temperature_data

logger = logging.getLogger(__name__)


def merge_and_resample(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Join dataframes on a daily frequency using the mean."""
    df = None
    for d in dfs:
        d = d.set_index('timestamp')
        df = d if df is None else df.join(d, how='inner')
    df = df.resample('D').mean().dropna().reset_index()
    return df


def run_three_way_regression(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS model of movement against rainfall and temperature."""
    X = df[['rainfall_mm', 'temperature_C']]
    X = sm.add_constant(X)
    y = df['movement_mm']
    model = sm.OLS(y, X).fit()
    logger.info("\n%s", model.summary())
    ci90 = model.conf_int(alpha=0.10)
    logger.info("90%% confidence intervals:\n%s", ci90)
    return model


def compute_partial_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute partial correlations using pingouin."""
    p1 = partial_corr(data=df, x='rainfall_mm', y='movement_mm', covar='temperature_C')
    p2 = partial_corr(data=df, x='temperature_C', y='movement_mm', covar='rainfall_mm')
    return pd.concat([p1, p2], axis=0)


def analyze_movement_rain_temp(output_dir: Path = Path("analysis_outputs")) -> None:
    """Run three-way analysis of movement, rainfall and temperature.

    Parameters
    ----------
    output_dir : Path, optional
        Directory where result plots will be written. Created if it does not
        already exist.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info("Loading data...")
    df_mov = load_monitoring_data()
    df_rain = load_rainfall_data()
    df_temp = load_temperature_data()

    logger.info("Merging and resampling data...")
    df = merge_and_resample([df_mov, df_rain, df_temp])

    # Save merged data for interactive plotting
    (output_dir / "merged_data.json").write_text(
        df.to_json(orient="split", date_format="iso")
    )

    logger.info("Running regression analysis...")
    model = run_three_way_regression(df)

    logger.info("Computing partial correlations...")
    pcorrs = compute_partial_correlations(df)
    logger.info("Partial correlations:\n%s", pcorrs)

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
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    logger.info("Generating plots...")
    pd.plotting.scatter_matrix(df[['movement_mm', 'rainfall_mm', 'temperature_C']], diagonal='kde')
    plt.suptitle("Scatter Matrix")
    plt.savefig(output_dir / 'scatter_matrix.png')
    plt.clf()

    from statsmodels.tsa.stattools import ccf

    def compute_ccf(x, y) -> list:
        """Return first 30 values of the cross-correlation function."""
        return ccf(x, y)[:30].tolist()

    rainfall_ccf = compute_ccf(df['rainfall_mm'], df['movement_mm'])
    temperature_ccf = compute_ccf(df['temperature_C'], df['movement_mm'])

    (output_dir / "rainfall_ccf.json").write_text(json.dumps(rainfall_ccf))
    (output_dir / "temperature_ccf.json").write_text(json.dumps(temperature_ccf))

    def plot_ccf(values, name):
        plt.bar(range(len(values)), values)
        plt.title(f"{name} → Movement CCF")
        plt.xlabel("Lag (days)")
        plt.ylabel("Correlation")
        plt.savefig(output_dir / f"{name.lower()}_ccf.png")
        plt.clf()

    plot_ccf(rainfall_ccf, name='Rainfall')
    plot_ccf(temperature_ccf, name='Temperature')

    logger.info("Analysis complete. Plots saved to %s", output_dir.resolve())

