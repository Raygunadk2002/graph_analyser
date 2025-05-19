import logging
from typing import List

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


def analyze_movement_rain_temp() -> None:
    """Run three-way analysis of movement, rainfall and temperature."""
    logger.info("Loading data...")
    df_mov = load_monitoring_data()
    df_rain = load_rainfall_data()
    df_temp = load_temperature_data()

    logger.info("Merging and resampling data...")
    df = merge_and_resample([df_mov, df_rain, df_temp])

    logger.info("Running regression analysis...")
    model = run_three_way_regression(df)

    logger.info("Computing partial correlations...")
    pcorrs = compute_partial_correlations(df)
    logger.info("Partial correlations:\n%s", pcorrs)

    logger.info("Generating plots...")
    pd.plotting.scatter_matrix(df[['movement_mm', 'rainfall_mm', 'temperature_C']], diagonal='kde')
    plt.suptitle("Scatter Matrix")
    plt.savefig('scatter_matrix.png')
    plt.clf()

    from statsmodels.tsa.stattools import ccf

    def plot_ccf(x, y, name):
        c = ccf(x, y)[:30]
        plt.bar(range(len(c)), c)
        plt.title(f"{name} → Movement CCF")
        plt.xlabel("Lag (days)")
        plt.ylabel("Correlation")
        plt.savefig(f"{name.lower()}_ccf.png")
        plt.clf()

    plot_ccf(df['rainfall_mm'], df['movement_mm'], name='Rainfall')
    plot_ccf(df['temperature_C'], df['movement_mm'], name='Temperature')

    logger.info("Analysis complete. Plots saved to disk.")

