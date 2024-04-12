from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb  # Add to requirements.txt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import mlflow
from panda_utils import time_series_train_val_test_split, split_features_and_labels


mlflow.autolog()
mlflow.set_experiment('Energy Use Forecasting')
make_exploration_plots = False
make_feature_plots = False
make_validation_lots = True
plt.style.use('fivethirtyeight')
model_types = ['xgboost']  # TODO: add svm
for model_type in model_types:
    if model_type not in ["xgboost"]:
        raise NotImplementedError(f'Unsupported model type {model_type}.')


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

    df = create_time_unit_features(df)

    features = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
    target = 'MW'

    train, val, test = (
        time_series_train_val_test_split(df, val_ratio=0.15, test_ratio=0.15))
    # Features separate from targets from now on
    X_train, y_train, X_val, y_val, X_test, y_test = split_features_and_labels(train, val, test=test,
                                                                               features=features, target=target)
    # Avoid accidentally bleeding test/val info
    del df

    if make_exploration_plots:
        explore_trainval_relationships(train, val)
        explore_seasonality(train)

    for model_type in model_types:
        with mlflow.start_run() as run:
            print(f"Starting run: {run.info.run_id}")
            model, parameters = configure_model(model_type)

            clf, model = train_select_model(model=model, parameters=parameters, X_train=X_train, y_train=y_train)

            # if model_type == 'xgboost':
            #     print(f"Optimal number of trees: {model.best_iteration}")
            #     if make_feature_plots:
            #         plot_trainval_results(results, best_iteration=model.best_iteration)

            if make_feature_plots and model_type == 'xgboost':
                fi = pd.DataFrame(data=model.feature_importances_,
                                  index=model.feature_names_in_,
                                  columns=['importance'])
                ax = fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
                mlflow.log_figure(ax.get_figure(), 'feature_importance.png')
                plt.show()

            # Fitting is done, so add predictions back for plotting
            train['prediction'] = model.predict(X_train)
            val['prediction'] = model.predict(X_val)

            val, rmse = get_accuracy_metrics_df(val, target)
            print(f'RMSE of the best {model_type} model: {rmse}')
            print(f"Best model parameters")
            print(clf.best_params_)

            if make_validation_lots:
                # Predictions are equally bad on training and validation data - the model is underfitting
                # Specifically, it is unable to predict extremes
                trainval = pd.concat([train, val])
                if model_type == 'xgboost':
                    filename_trainval_preds = f"{model_type}_depth-{clf.best_params_['max_depth']}_rmse-{rmse}_lr-{clf.best_params_['learning_rate']}"
                fig_trainval_preds = plot_trainval_preds(trainval, target,
                                                         save_file=models_path / (filename_trainval_preds + ".png"))
                mlflow.log_figure(fig_trainval_preds, 'trainval_predictions.png')
                # plot_trainval_preds_week(df)


def train_select_model(model, parameters, X_train, y_train):
    clf = GridSearchCV(model, parameters, scoring='neg_root_mean_squared_error', cv=3, verbose=3)
    clf.fit(X_train, y_train)
    # print(sorted(clf.cv_results_.keys()))
    model = clf.best_estimator_
    return clf, model


def configure_model(model_type):
    if model_type == 'xgboost':
        max_depths = [5, 10, 15, 20, 30, 50]
        learning_rates = [1, 0.3, 1e-1, 1e-2, 1e-3, 1e-4]
        parameters = {'booster': ['gbtree'],
                      'max_depth': max_depths, 'objective': ['reg:squarederror'],
                      'learning_rate': learning_rates}
        model = xgb.XGBRegressor(base_score=0.5, n_estimators=10000)
    return model, parameters


def get_accuracy_metrics_df(df, target: str):
    rmse = np.sqrt(mean_squared_error(df['prediction'], df[target]))
    rmse = round(rmse, 2)
    df['error'] = np.abs(df[target] - df['prediction'])
    df['percent_error'] = np.abs(df[target] - df['prediction']) / df[target]
    df['date'] = df.index.date
    df.groupby(['date'])['error'].mean().sort_values(ascending=False).head(10)
    return df, rmse


def plot_trainval_preds(df, target, save_file=None, display=None):
    ax = df[target].plot(figsize=(15, 5))
    df['prediction'].plot(ax=ax, style='.')
    plt.legend(['Truth Data', 'Predictions'])
    plt.axvline(df, color="gray", lw=3, label=f"Val split point")
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


def explore_trainval_relationships(train, val):
    # The plot shows no major discrepancy between trends in train and val data. No need to correct for these
    fig, ax = plt.subplots(figsize=(15, 5))
    train.plot(ax=ax, label='Training Set', title='Train/Val Data Split')
    val.plot(ax=ax, label='Validation Set')
    ax.axvline('01-01-2015', color='black', ls='--')
    ax.legend(['Training Set', 'Validation Set'])
    plt.show()
    train.loc[(train.index > '01-01-2010') & (train.index < '01-08-2010')] \
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
