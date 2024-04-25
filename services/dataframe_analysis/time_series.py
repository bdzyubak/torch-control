import mlflow
import numpy as np
import pandas as pd


def create_time_unit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time series features based on time series index.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError('Only Dataframe inputs are supported')

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError('This function only applies to dataframes index with pd.DatetimeIndex')

    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df


def get_accuracy_metrics_df(df_pred: pd.DataFrame, y: pd.Series, split: str = '', log_metrics_to_mlflow=False):
    # Per datapoint metrics
    error = y - df_pred['prediction']
    percent_error = (y - df_pred['prediction']) / y * 100
    df_pred['error'] = error
    df_pred['percent_error'] = percent_error
    df_pred['abs_error'] = df_pred['error'].abs()
    df_pred['abs_percent_error'] = df_pred['percent_error'].abs()
    df_pred['date'] = df_pred.index.date

    # Summary metrics
    rmse = round(np.sqrt(df_pred['abs_error'].pow(2).mean()))
    mae = round(df_pred['abs_error'].mean())
    mape = round(df_pred['abs_percent_error'].mean(), 1)

    me = df_pred['error'].mean().round()
    mpe = df_pred['percent_error'].mean().round(1)

    if log_metrics_to_mlflow:
        mlflow.log_metric(split + '_rmse', rmse)
        mlflow.log_metric(split + '_mae', mae)
        mlflow.log_metric(split + '_mape', mape)
        mlflow.log_metric(split + '_me', me)
        mlflow.log_metric(split + '_mpe', mpe)

    print(f'{split} Root Mean Squared Error: {rmse} MW, Mean Absolute Error: {mae} MW, '
          f'Mean Absolute Percent Error: {mape}%')
    print(f'{split} Mean Error: {me} MW, Mean Percent Error: {mpe}%')
    return df_pred, rmse
