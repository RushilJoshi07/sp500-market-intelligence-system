"""
Data loader for S&P 500 Market Intelligence Dashboard.
Loads all pre-computed CSV files for dashboard display.
"""
import os
import pandas as pd


# Path to dashboard data directory.
DASHBOARD_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "dashboard",
)


def load_sp500() -> pd.DataFrame:
    """
    Load S&P 500 price history.

    :return: DataFrame with S&P 500 OHLCV data
    """
    df = pd.read_csv(
        os.path.join(DASHBOARD_DATA_DIR, "sp500.csv"),
        index_col=0,
        parse_dates=True,
    )
    return df


def load_sectors() -> pd.DataFrame:
    """
    Load sector ETF price history.

    :return: DataFrame with sector ETF closing prices
    """
    df = pd.read_csv(
        os.path.join(DASHBOARD_DATA_DIR, "sectors.csv"),
        index_col=0,
        parse_dates=True,
    )
    return df


def load_macro_daily() -> pd.DataFrame:
    """
    Load daily macro indicators.

    :return: DataFrame with daily macro data
    """
    df = pd.read_csv(
        os.path.join(DASHBOARD_DATA_DIR, "macro_daily.csv"),
        index_col=0,
        parse_dates=True,
    )
    return df


def load_regime_labels() -> pd.Series:
    """
    Load market regime labels per trading day.

    :return: Series with regime label per date
    """
    df = pd.read_csv(
        os.path.join(DASHBOARD_DATA_DIR, "regime_labels.csv"),
        index_col=0,
        parse_dates=True,
    )
    return df.squeeze()


def load_regime_stats() -> pd.DataFrame:
    """
    Load regime statistics.

    :return: DataFrame with stats per regime
    """
    return pd.read_csv(
        os.path.join(DASHBOARD_DATA_DIR, "regime_stats.csv")
    )


def load_weekly_recommendations() -> pd.DataFrame:
    """
    Load pre-computed weekly sector recommendations.

    :return: DataFrame with weekly scores and recommendations
    """
    df = pd.read_csv(
        os.path.join(
            DASHBOARD_DATA_DIR, "weekly_recommendations.csv"
        ),
        index_col=0,
        parse_dates=True,
    )
    return df


def load_attribution_df() -> pd.DataFrame:
    """
    Load regime attribution data.

    :return: DataFrame with sector returns per regime period
    """
    return pd.read_csv(
        os.path.join(DASHBOARD_DATA_DIR, "attribution_df.csv")
    )


def load_walk_forward_results() -> pd.DataFrame:
    """
    Load walk forward validation results.

    :return: DataFrame with MAPE and direction per window
    """
    return pd.read_csv(
        os.path.join(
            DASHBOARD_DATA_DIR, "walk_forward_results.csv"
        )
    )


def load_all() -> dict:
    """
    Load all dashboard data in one call.

    :return: dictionary with all DataFrames
    """
    return {
        "sp500"                 : load_sp500(),
        "sectors"               : load_sectors(),
        "macro_daily"           : load_macro_daily(),
        "regime_labels"         : load_regime_labels(),
        "regime_stats"          : load_regime_stats(),
        "weekly_recommendations": load_weekly_recommendations(),
        "attribution_df"        : load_attribution_df(),
        "walk_forward_results"  : load_walk_forward_results(),
    }