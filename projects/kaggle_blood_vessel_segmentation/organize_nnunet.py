from pathlib import Path
import shutil
import os
import json
import argparse

from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess


def main(args):
    dataset_name = 'BloodVesselSegmentation'
    dataset_id = 501
    dataset_name_id = '_'.join([str(dataset_id), dataset_name])
    extension, file_channel_suffix, path_origin, path_target = _setup_paths(dataset_name_id)

    if args.copy_data:
        for data_type in ['train', 'test']:
            case_index = 0
            if data_type == 'train':
                output_suffix = 'Tr'
                input_types = ['images', 'labels']
            else:
                output_suffix = 'Ts'
                input_types = ['images']

            path_origin_traintest = path_origin / data_type
            subdirs = [name for name in list(path_origin_traintest.glob('*')) if name.is_dir()]
            subdirs = [name for name in subdirs if name.stem != 'kidney_3_dense']  # Drop corrupt dataset with no images

            for subdir in subdirs:
                print(f'Copying files for acquisition: {subdir.stem}')

                for file_type in input_types:
                    case_index = _copy_files_images_labels(case_index, extension, file_channel_suffix, file_type,
                                                           output_suffix, path_target, subdir)

            if data_type == 'train':
                _save_dataset_json(path_target=path_target, num_training=int(case_index/2), extension=extension)
            print(f'Done copying files.')

    if args.run_preprocessing:
        plan_and_preprocess(dataset_id=dataset_id, check_dataset_integrity=True,
                            configurations_to_run=args.configurations)


def _setup_paths(dataset_name_id):
    path_origin = Path(r"D:\data\blood-vessel-segmentation")
    nnUnet_raw = Path(os.environ['nnUnet_raw'])
    extension = '.tif'
    file_channel_suffix = '0000'
    path_target = nnUnet_raw / ('Dataset' + dataset_name_id)
    if path_target.exists():
        shutil.rmtree(path_target)
    path_target.mkdir(parents=True)
    return extension, file_channel_suffix, path_origin, path_target


def _copy_files_images_labels(case_index, extension, file_channel_suffix, file_type, output_suffix, path_target,
                              subdir):
    path_target_full = path_target / (file_type + output_suffix)
    if not path_target_full.exists():
        path_target_full.mkdir()
    path_origin_full = subdir / file_type
    files_origin = list(path_origin_full.glob('*' + extension))
    if not files_origin:
        raise OSError(f'Files not found at origin dir: {path_origin_full}')
    case_index = _copy_files_for_scan_and_type(case_index, extension, file_channel_suffix, files_origin,
                                               path_target_full)
    return case_index


def _copy_files_for_scan_and_type(case_index, extension, file_channel_suffix, files_origin, path_target_full):
    num_files_origin = len(files_origin)
    for ind, file_origin in enumerate(files_origin):
        file_target = path_target_full / ('BVsegm_' + str(case_index).zfill(4) + '_'
                                          + file_channel_suffix + extension)
        if ind % 100 == 0:
            print(f'{ind + 1}/{num_files_origin}: {file_origin} to {file_target}')
        shutil.copy(file_origin, file_target)
        case_index += 1
    return case_index


def _save_dataset_json(path_target, num_training, extension):
    data_info = {
        "channel_names": {
            "0": "microscopy",
        },
        "labels": {
            "background": 0,
            "vessel": 1,
        },
        "numTraining": num_training,
        "file_ending": extension,
    }
    path_target_json = path_target / 'dataset.json'
    with open(path_target_json, 'w') as f:
        json.dump(data_info, f)
    print(f'Saved json: {path_target_json}')


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--copy_data', action='store_true',
                        help="Download and copy data to nnUnet_raw")
    parser.add_argument('--run_preprocessing', action='store_true',
                        help="Run preprocessing. Should only be done once, when data is downloaded.")
    parser.add_argument('--configurations', required=False, default=['2d', '3d_fullres', '3d_lowres'],
                        nargs='+',
                        help='[OPTIONAL] Configurations for which the preprocessing should be run. Default: 2d 3d_fullres '
                             '3d_lowres. 3d_cascade_fullres does not need to be specified because it uses the data '
                             'from 3d_fullres. Configurations that do not exist for some dataset will be skipped.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_command_line_args()

    main(args)
