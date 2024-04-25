from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from panda_utils import time_series_train_val_test_split, split_features_and_labels_train_val
from services.dataframe_analysis.time_series import create_time_unit_features


def get_energy_use_data(make_exploration_plots=False):
    color_pal = sns.color_palette()
    input_file = Path(r'D:\data\ML\PowerConsumption\AEP_hourly.csv')
    df = pd.read_csv(input_file)
    df.drop_duplicates(subset="Datetime", inplace=True)
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
    X_train, y_train, X_val, y_val, X_test, y_test = split_features_and_labels_train_val(train, val, test=test,
                                                                                         features=features,
                                                                                         target=target)
    if make_exploration_plots:
        explore_trainval_relationships(train, val)
        explore_seasonality(train)

    # Return as dict to avoid screwing up order and polluting train/val/test splits
    # For regulated use, test set should be totally separate and not visible to devs
    data = dict()
    data['X_train'] = X_train
    data['y_train'] = y_train
    data['X_val'] = X_val
    data['y_val'] = y_val
    data['X_test'] = X_test
    data['y_test'] = y_test
    data['target'] = target

    return data


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
