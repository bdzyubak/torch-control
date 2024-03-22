import pytest
import pandas as pd

from utils.panda_utils import pick_columns_trim_name


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


if __name__ == '__main__':
    retcode = pytest.main(__file__)
    if retcode == 0:
        print('All tests passed.')
