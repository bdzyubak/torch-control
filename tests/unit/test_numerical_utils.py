from numerical_utils import scientific_notation


def test_scientific_notation():
    rmse = 0.47568238013868186
    mae = 0.0040113019931441495
    mape = 2.8670413040001215e-07

    result = scientific_notation([rmse, mae, mape])

    assert result == ('4.76e-01', '4.01e-03', '2.87e-07')

    rmse = 0.47568238013868186

    result = scientific_notation(rmse)

    assert result == ('4.76e-01', '4.01e-03', '2.87e-07')


def test_round_numbers():
    rmse = 0.47568238013868186
    mae = 0.0040113019931441495
    mape = 2.8670413040001215e-07

    result = scientific_notation([rmse, mae, mape])

    assert result == ('4.76e-01', '4.01e-03', '2.87e-07')

    rmse = 0.47568238013868186

    result = scientific_notation(rmse)

    assert result == ('4.76e-01', '4.01e-03', '2.87e-07')
