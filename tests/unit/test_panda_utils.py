import pytest
import pandas as pd
import numpy as np

from utils.panda_utils import (pick_columns_trim_name, is_close, time_series_train_val_test_split,
                               split_features_and_labels)


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
    assert list(train.index()) == [0, 1, 2, 3, 4, 5] and list(val.index()) == [6, 7] and list(train.index()) == [8, 9]
    # TODO: add a couple of other scenarios to test generalizability


# def split_features_and_labels():
#     result = is_close(1, 1.0001)
#     assert result is True, "Failed to confirm similarity default tolerance"
#
#     result = is_close(1, 1.001)
#     assert result is False, "Failed to flag difference with default tolerance"
#
#     result = is_close(1989, 1990, abs_tol=1)
#     assert result is True, "Failed to confirm similarity specified tolerance"
#
#     result = is_close(1989, 1991, abs_tol=1)
#     assert result is False, "Failed to flag difference with specified tolerance"


if __name__ == '__main__':
    retcode = pytest.main(__file__)
    if retcode == 0:
        print('All tests passed.')
