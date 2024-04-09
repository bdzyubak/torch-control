import pytest
from utils.torch_utils import get_model_size_mb, get_model_param_num, get_model_size
from utils.panda_utils import is_close
from torchvision.models import resnet18


def test_filename_to_title():
    model = resnet18()
    model_size_mb = get_model_size_mb(model)
    assert is_close(model_size_mb, 44.6), 'Model size calculated incorrectly.'

    model_param_total, model_param_trainable = get_model_param_num(model)
    assert model_param_total == 15 and model_param_trainable == 15, 'Model parameters calculated incorrectly'

    size_all_mb, model_param_total, model_param_trainable = get_model_size(model)
    assert is_close(size_all_mb, 44.6) and model_param_total == 15 and model_param_trainable == 15, \
        'Model size and parameter number together calculated incorrectly'


if __name__ == '__main__':
    retcode = pytest.main(__file__)
    if retcode == 0:
        print('All tests passed.')
