import pytest
import pandas as pd
import numpy as np

from utils.panda_utils import (pick_columns_trim_name, is_close, time_series_train_val_test_split,
                               split_features_and_labels, split_features_and_labels_train_val)


def test_filename_to_title():
    df = pd.DataFrame(columns=['Accuracy: 0.5 thresh', 'Unlabeled: 0.6 thresh', 'Accuracy: 0.7 thresh', 'Unlabeled: 0.8 thresh'])
    result = pick_columns_trim_name(df, 'Accuracy: ')
    expected_result = pd.DataFrame(columns=['0.5 thresh', '0.5 thresh', 'Accuracy: 0.7', 'Unlabeled: 0.7'])
    assert result == expected_result, "Failed to pick and replace the right columns"

    df = pd.DataFrame(
        columns=['Accuracy: 0.5 thresh', 'Unlabeled: 0.6 thresh', 'Accuracy: 0.7 thresh', 'Unlabeled: 0.8 thresh'])
    result = pick_columns_trim_name(df, 'Unlabeled:')
    expected_result = pd.DataFrame(columns=['0.6 thresh', '0.5 thresh', 'Accuracy: 0.7', 'Unlabeled: 0.8'])
    assert result == expected_result, "Failed to trim spaces"


def test_is_close():
    result = is_close(1, 1.0001)
    assert result is True, "Failed to confirm similarity default tolerance"

    result = is_close(1, 1.001)
    assert result is False, "Failed to flag difference with default tolerance"

    result = is_close(1989, 1990, abs_tol=1)
    assert result is True, "Failed to confirm similarity specified tolerance"

    result = is_close(1989, 1991, abs_tol=1)
    assert result is False, "Failed to flag difference with specified tolerance"


def test_time_series_train_val_test_split():
    df = pd.DataFrame(index=np.arange(10), columns=['Data1', 'Data2'])
    train, val, test = time_series_train_val_test_split(df, val_ratio=0.3, test_ratio=0.3)
    assert list(train.index) == [0, 1, 2, 3] and list(val.index) == [4, 5, 6] and list(test.index) == [7, 8, 9]

    train, val, test = time_series_train_val_test_split(df, val_ratio=0.2, test_ratio=0.2)
    assert list(train.index) == [0, 1, 2, 3, 4, 5] and list(val.index) == [6, 7] and list(test.index) == [8, 9]

    train, val, test = time_series_train_val_test_split(df, val_ratio=0.2)
    assert list(train.index) == [0, 1, 2, 3, 4, 5, 6, 7] and list(val.index) == [8, 9] and test.empty


def test_split_features_and_labels():
    df = pd.DataFrame(index=np.arange(10), columns=['X1', 'X2', 'Dont Need', 'y'])
    features = ['X1', 'X2']
    target = 'y'
    X, y = split_features_and_labels(df, features=features, target=target)
    assert list(X.columns) == features and y.name == target


def split_features_and_labels_train_val():
    train = pd.DataFrame(index=np.arange(2), columns=['X1', 'X2', 'Dont Need', 'y'])
    val = pd.DataFrame(index=np.arange(3, 5), columns=['X1', 'X2', 'Dont Need', 'y'])
    features = ['X1', 'X2']
    target = 'y'
    X_train, y_train, X_val, y_val, X_test, y_test = split_features_and_labels_train_val(train, val, features=features,
                                                                                         target=target)
    assert list(X_train.columns) == features and list(X_val.columns) == features and X_test is None

    assert y_train.name == target and y_val.name == target and y_test is None

    assert list(X_train.index) == [0, 1] and list(X_val.index) == [2, 3]

    train = pd.DataFrame(index=np.arange(2), columns=['X1', 'X2', 'Dont Need', 'y'])
    val = pd.DataFrame(index=np.arange(3, 5), columns=['X1', 'X2', 'Dont Need', 'y'])
    test = pd.DataFrame(index=np.arange(5, 7), columns=['X1', 'X2', 'Dont Need', 'y'])
    features = ['X1', 'X2']
    target = 'y'
    X_train, y_train, X_val, y_val, X_test, y_test = split_features_and_labels_train_val(train, val, test=test,
                                                                                         features=features,
                                                                                         target=target)
    assert list(X_train.columns) == features and list(X_val.columns) == features and list(X_test.columns) == features

    assert y_train.name == target and y_val.name == target and y_test.name == target

    assert list(X_train.index) == [0, 1] and list(X_val.index) == [2, 3] and list(X_test.index) == [4, 5]


if __name__ == '__main__':
    retcode = pytest.main(__file__)
    if retcode == 0:
        print('All tests passed.')
