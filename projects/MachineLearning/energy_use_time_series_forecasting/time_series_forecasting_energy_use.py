from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import mlflow
import sklearn
from os_utils import get_memory_use
from panda_utils import set_display_rows_cols
from projects.MachineLearning.energy_use_time_series_forecasting.dataloader import get_energy_use_data
from services.dataframe_analysis.time_series import get_accuracy_metrics_df

set_display_rows_cols()
mlflow.autolog()
mlflow.set_experiment('Energy Use Forecasting')
make_exploration_plots = False
make_feature_plots = False
make_validation_lots = True
plt.style.use('fivethirtyeight')
model_types = ['xgboost']
for model_type in model_types:
    if model_type not in ["xgboost", "svm"]:
        raise NotImplementedError(f'Unsupported model type {model_type}.')


def main():
    data = get_energy_use_data(make_exploration_plots)

    models_path = Path(r'D:\Models\ML') / Path(__file__).stem
    models_path.mkdir(parents=True, exist_ok=True)

    train_validate_model(data['X_train'], data['y_train'], data['X_val'], data['y_val'], models_path,
                         model_types=model_types)


def train_validate_model(X_train, y_train, X_val, y_val, models_path, model_types=None):
    if model_types is None:
        model_types = ['xgboost']

    for model_type in model_types:
        if model_type == 'xgboost':
            clf = xgb.XGBRegressor()
            # TODO: Chain these optimizations procedurally. Presently, the blocks need to be uncommented and best
            #  parameters input into the subsequent block
            # For each parameter where the best setting is the default - drop the parameter for simplicity
            # run_name = f'Optimize {model_type} depth/estimators/lr'
            # params = {'learning_rate': [1, 0.3, 1e-1, 1e-2], 'n_estimators': [100, 300, 500, 1000],
            #           'max_depth': [3, 5, 10, 20]}
            #
            # run_name = f'Optimize {model_type} col_sample/subsample/min_child_weight'
            # params = {'learning_rate': [1e-2], 'max_depth': [5], 'n_estimators': [300], 'subsample': [1, 0.5, 0.3, 0.1],
            #           'colsample_bytree': [1, 0.5, 0.3, 0.1], 'min_child_weight': range(1, 20, 2)}
            # {'colsample_bytree': 1, 'learning_rate': 0.01, 'max_depth': 5, 'min_child_weight': 7, 'n_estimators': 300,
            #  'subsample': 0.1}

            # run_name = f'Optimize {model_type} child_weight'
            # params = {'learning_rate': [1], 'max_depth': [3], 'min_child_weight': range(1, 6, 2)}

            run_name = f'Optimal {model_type} validate'
            params = {'learning_rate': [0.01], 'max_depth': [20], 'n_estimators': [300]}

        elif model_type == 'svm':
            clf = sklearn.svm.SVR()
            run_name = f'Optimize {model_type} col_sample/subsample/min_child_weight'
            params = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                      'kernel': ['rbf']}
        else:
            raise NotImplementedError(model_type)

        with (mlflow.start_run(run_name=run_name) as parent_run):
            xgb_grid = GridSearchCV(clf, params, scoring='neg_root_mean_squared_error', cv=2, verbose=3)
            get_memory_use(code_point='Pre training')
            xgb_grid.fit(X_train, y_train)
            get_memory_use(code_point='Post training', log_to_mlflow=True)

            print(f"Best parameters: {xgb_grid.best_params_}")
            model = xgb.XGBRegressor(**xgb_grid.best_params_)
            model.fit(X_train, y_train)

            if make_feature_plots and model_type == 'xgboost':
                fi = pd.DataFrame(data=model.feature_importances_,
                                  index=model.feature_names_in_,
                                  columns=['importance'])
                ax = fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
                mlflow.log_figure(ax.get_figure(), 'feature_importance.png')
                plt.show()

            # Fitting is done, so add predictions back for plotting
            X_train['prediction'] = model.predict(X_train)
            X_val['prediction'] = model.predict(X_val)

            train, train_rmse = get_accuracy_metrics_df(X_train, y_train, split='train', log_metrics_to_mlflow=True)
            val, val_rmse = get_accuracy_metrics_df(X_val, y_val, split='val', log_metrics_to_mlflow=True)

            print(f"Train RMSE {train_rmse}, Val RMSE {val_rmse}")

            if make_validation_lots:
                # Predictions are equally bad on training and validation data - the model is underfitting
                # Specifically, it is unable to predict extremes

                filename = f"{model_type}_{val_rmse}_train_predictions"
                fig_trainval_preds = plot_trainval_preds(X_train, y_train, val_split_index=train.index.max(),
                                                         save_file=models_path / (filename + ".png"))
                mlflow.log_figure(fig_trainval_preds, filename + '.png')

                filename = f"{model_type}_{val_rmse}_val_predictions"
                fig_trainval_preds = plot_trainval_preds(X_val, y_val, val_split_index=train.index.max(),
                                                         save_file=models_path / (filename + ".png"))
                mlflow.log_figure(fig_trainval_preds, filename + '.png')
                plt.close(fig_trainval_preds)


def plot_trainval_preds(X, y, val_split_index=None, save_file=None, display=None):
    ax = y.plot(figsize=(15, 5))
    X['prediction'].plot(ax=ax, style='.')
    plt.legend(['Truth Data', 'Predictions'])
    if val_split_index is not None:
        # Use this for plotting both train and val on the same figure.
        # Warning, this can visually make you extend good train predictions to get the impression val is better than
        # it is
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


if __name__ == '__main__':
    main()
