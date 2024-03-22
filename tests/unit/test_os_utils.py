import pytest
from utils.os_utils import filename_to_title
from pathlib import Path


def test_filename_to_title():
    file_path = r'D:\ValidationResults\experiment.csv'
    expected_result = 'Experiment'
    assert filename_to_title(file_path) == expected_result, "Failed to convert full path with one word"

    file_path = r'D:\ValidationResults\experiment_name.csv'
    expected_result = 'Experiment Name'
    assert filename_to_title(file_path) == expected_result, "Failed to convert full path with underscores"

    file_path = r'D:\ValidationResults\experiment name.csv'
    expected_result = 'Experiment Name'
    assert filename_to_title(file_path) == expected_result, "Failed to convert full path with spaces"

    file_path = r'/home/user/experiment name.csv'
    expected_result = 'Experiment Name'
    assert filename_to_title(file_path) == expected_result, "Failed to conver Linux path"

    file_path = r'D:\ValidationResults\LaLa la La.csv'
    expected_result = 'LaLa La La'
    assert filename_to_title(file_path) == expected_result, "Failed to preserve existing capitalization"

    file_path = r'D:\ValidationResults\_experiment_name.csv'
    expected_result = 'Experiment Name'
    assert filename_to_title(file_path) == expected_result, "Failed to deal with prefixes"

    file_path = 'experiment name.csv'
    expected_result = 'Experiment Name'
    assert filename_to_title(file_path) == expected_result, "Failed to deal with raw filename"

    file_path = 'experiment name'
    expected_result = 'Experiment Name'
    assert filename_to_title(file_path) == expected_result, "Failed to deal with raw filename with no extension"

    file_path = r'D:\ValidationResults\experiment.csv'
    expected_result = 'Experiment Name'
    assert filename_to_title(file_path) == expected_result, "Failed to deal with Path"


if __name__ == '__main__':
    retcode = pytest.main(__file__)
    if retcode == 0:
        print('All tests passed.')
