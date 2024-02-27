from pathlib import Path
import shutil
import os
import json


path_origin = Path(r"D:\data\blood-vessel-segmentation")
nnUnet_raw = Path(os.environ['nnUnet_raw'])
extension = '.tif'

data_info = {
     "channel_names": {
       "0": "microscopy",
     },
     "labels": {
       "background": 0,
       "vessel": 1,
     },
     "numTraining": 32,
     "file_ending": extension,
    }


for imaging_type in ['kidney_1_dense', 'kidney_1_voi']:
    file_channel_suffix = '0000'
    if imaging_type.endswith('dense'):
        path_target = nnUnet_raw / 'Dataset501_BloodVesselSegmentationDense'
    else:
        path_target = nnUnet_raw / 'Dataset502_BloodVesselSegmentationVOI'

    if path_target.exists():
        shutil.rmtree(path_target)

    path_target.mkdir(parents=True)
    path_target_json = path_target / 'dataset.json'
    with open(path_target_json, 'w') as f:
        json.dump(data_info, f)
    print(f'Saved json: {path_target_json}')

    print(f'Copying files to: {path_target}')
    for data_type in ['train', 'test']:
        if data_type == 'train':
            output_suffix = 'Tr'
        else:
            output_suffix = 'Ts'
        input_types = ['images']

        for file_type in ['images', 'labels']:
            path_target_full = path_target / (file_type + output_suffix)
            if not path_target_full.exists():
                path_target_full.mkdir()

            path_origin_full = path_origin / data_type / imaging_type / file_type
            files_origin = list(path_origin_full.glob('*' + extension))
            if not files_origin:
                raise OSError(f'Files not found at origin dir: {path_origin_full}')

            num_files_origin = len(files_origin)
            for ind, file_origin in enumerate(files_origin):
                file_target = path_target_full / ('BVsegm_' + file_origin.stem + '_'
                                                  + file_channel_suffix + extension)
                if ind % 100 == 0:
                    print(f'{ind + 1}/{num_files_origin}: {file_origin} to {file_target}')
                shutil.copy(file_origin, file_target)

    print(f'Done copying files.')
