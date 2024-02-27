import pandas as pd
from pathlib import Path
import opendatasets as od
import shutil


# Log into Kaggle and generate access token. You must also accept the competition rules.
# The token will be read automatically if it is in the same directory as this file, but for security reasons, it cannot
# be part of the source code. So, copy the credentials manually, since this is a one-time deal
dataset = 'https://www.kaggle.com/competitions/blood-vessel-segmentation/data'
path_origin = Path(__file__).parent / 'blood-vessel-segmentation'
path_origin_zipfile = path_origin / 'blood-vessel-segmentation.zip'  # Dataset specific. Enter manually


path_target = 'D:\\data\\blood-vessel-segmentation'
run_download = False


if run_download:
    od.download(dataset)
else:
    if not Path.exists(path_origin):
        raise OSError('Data not downloaded. Flip run_download toggle to True.')
    od.utils.archive.extract_archive(from_path=str(path_origin), to_path=str(path_target))

path_origin_zipfile.unlink()
Path.rename(path_origin, path_target)  # copy
