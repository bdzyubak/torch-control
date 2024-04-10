from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb  # Add to requirements.txt
from sklearn.metrics import mean_squared_error
import mlflow

from utils.plotting import set_plotting_defaults  # Keep this line. Sets better plotting on import

mlflow.autolog()
mlflow.set_experiment('Energy Use Forecasting')
make_exploration_plots = False
make_feature_plots = False
make_validation_lots = True
val_split_index = '01-01-2015'
plt.style.use('fivethirtyeight')
learning_rate = 0.001
# max_depth_search = [5, 10, 15, 30, 50]
max_depth_search = [10]


def main():
    color_pal = sns.color_palette()
    input_file = Path(r'D:\data\ML\PowerConsumption\AEP_hourly.csv')
    models_path = Path(r'D:\Models\ML') / Path(__file__).stem
    models_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)
    df = df.set_index('Datetime')
    df.index = pd.to_datetime(df.index)

    data_source = input_file.stem.split('_')[0]
    df = df.rename(columns={data_source + '_MW': 'MW'})

    if make_exploration_plots:
        df.plot(style='.',
                figsize=(15, 5),
                color=color_pal[0],
                title='Energy Use in MW')
        plt.show()

    train, val = train_val_split(df)

    if make_exploration_plots:
        explore_trainval_relationships(df, train, val)

    df = create_time_unit_features(df)
    # Redo trainval split with new features
    train, val = train_val_split(df)

    if make_exploration_plots:
        explore_seasonality(df)

    features = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
    target = 'MW'

    X_train = train[features]
    y_train = train[target]

    X_val = val[features]
    y_val = val[target]

    with mlflow.start_run() as run:
        for max_depth in max_depth_search:

            df_exp = df.copy()
            model = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                                     n_estimators=10000,
                                     early_stopping_rounds=50,
                                     objective='reg:squarederror',
                                     max_depth=max_depth,
                                     learning_rate=learning_rate)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_val, y_val)],
                      verbose=100)
            print(f"Optimal number of trees: {model.best_iteration}")

            results = model.evals_result()

            if make_feature_plots:
                plot_trainval_results(results, best_iteration=model.best_iteration)

            if make_feature_plots:
                fi = pd.DataFrame(data=model.feature_importances_,
                                  index=model.feature_names_in_,
                                  columns=['importance'])
                ax = fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
                mlflow.log_figure(ax.get_figure(), 'feature_importance.png')
                plt.show()

            # df = df.merge(val[['prediction']], how='left', left_index=True, right_index=True)
            trainval = df_exp[features]
            trainval['prediction'] = model.predict(trainval[features])
            df_exp = df_exp.merge(trainval[['prediction']], how='left', left_index=True, right_index=True)

            # val['prediction'] = model.predict(X_val)
            val = df_exp[df_exp.index >= val_split_index]
            rmse = get_accuracy_metrics(target, val)

            if make_validation_lots:
                # Predictions are equally bad on training and validation data - the model is underfitting
                # Specifically, it is unable to predict extremes
                fig_trainval_preds = plot_trainval_preds(df_exp,
                                                         save_file=models_path / f'xgboost_depth-{max_depth}_rmse-{rmse}_lr-{learning_rate}.png')
                mlflow.log_figure(fig_trainval_preds, 'trainval_predictions.png')
                # plot_trainval_preds_week(df)


def get_accuracy_metrics(target, val):
    rmse = np.sqrt(mean_squared_error(val['MW'], val['prediction']))
    rmse = round(rmse, 2)
    print(f'RMSE Score on Val set: {rmse}')
    val['error'] = np.abs(val[target] - val['prediction'])
    val['date'] = val.index.date
    val.groupby(['date'])['error'].mean().sort_values(ascending=False).head(10)
    return rmse


def plot_trainval_preds(df, save_file=None, display=None):
    ax = df[['MW']].plot(figsize=(15, 5))
    df['prediction'].plot(ax=ax, style='.')
    plt.legend(['Truth Data', 'Predictions'])
    plt.axvline(val_split_index, color="gray", lw=3, label=f"Val split point")
    ax.set_title('Raw Data and Prediction')

    if save_file is not None:
        plt.savefig(save_file)
        # By default, don't display when saving - assume hyperparameter search
        if display:
            plt.show()
    else:
        plt.show()
    return ax.get_figure()


def plot_trainval_preds_week(df):
    fig = plt.figure()
    ax = df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['MW'] \
        .plot(figsize=(15, 5), title='Week Of Data')
    df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['prediction'] \
        .plot(style='.')
    plt.legend(['Truth Data', 'Prediction'])
    plt.show()
    return fig


def plot_trainval_results(results, best_iteration=None):
    fig = plt.figure(figsize=(10, 7))
    plt.plot(results["validation_0"]["rmse"], label="Training loss")
    plt.plot(results["validation_1"]["rmse"], label="Validation loss")
    if best_iteration is not None:
        plt.axvline(best_iteration, color="gray", label=f"Optimal trees: {best_iteration}")
    plt.xlabel("Number of trees")
    plt.ylabel("Loss")
    plt.legend()
    return fig


def explore_trainval_relationships(df, train, val):
    # The plot shows no major discrepancy between trends in train and val data. No need to correct for these
    fig, ax = plt.subplots(figsize=(15, 5))
    train.plot(ax=ax, label='Training Set', title='Train/Val Data Split')
    val.plot(ax=ax, label='Validation Set')
    ax.axvline('01-01-2015', color='black', ls='--')
    ax.legend(['Training Set', 'Validation Set'])
    plt.show()
    df.loc[(df.index > '01-01-2010') & (df.index < '01-08-2010')] \
        .plot(figsize=(15, 5), title='Week Of Data')
    plt.show()


def explore_seasonality(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(data=df, x='hour', y='MW')
    ax.set_title('MW by Hour')
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(data=df, x='month', y='MW', palette='Blues')
    ax.set_title('MW by Month')
    plt.show()


def train_val_split(df):
    # For time series,  the train test split is being able to predict the future with the past,
    # not random. Gets 25% of data as val. Dataset contains data from other manufacturers, so we have a reserved test set
    train = df.loc[df.index < val_split_index]
    val = df.loc[df.index >= val_split_index]
    return train, val


def create_time_unit_features(df):
    """
    Create time series features based on time series index.
    """
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


if __name__ == '__main__':
    main()
