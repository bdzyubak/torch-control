from pathlib import Path
import shutil
import os
import json
import argparse
from PIL import Image
import numpy as np

from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.training.dataloading.utils import unpack_dataset
from nnunetv2.paths import nnUNet_preprocessed

from utils.os_utilities import get_file


def main(args):
    dataset_name = 'BloodVesselSegmentation'
    dataset_id = 501
    dataset_name_id = '_'.join([str(dataset_id), dataset_name])
    extension, file_channel_suffix, path_origin, path_target = _setup_paths(dataset_name_id)

    if not args.skip_copy:
        if path_target.exists():
            shutil.rmtree(path_target)
        path_target.mkdir(parents=True)

        for data_type in ['train', 'test']:
            if data_type == 'train':
                output_suffix = 'Tr'
            else:
                output_suffix = 'Ts'
            path_target_images = path_target / ('images' + output_suffix)
            if not path_target_images.exists():
                path_target_images.mkdir()
            path_target_labels = path_target / ('labels' + output_suffix)
            if not path_target_labels.exists():
                path_target_labels.mkdir()

            path_origin_traintest = path_origin / data_type
            subdirs = [name for name in list(path_origin_traintest.glob('*')) if name.is_dir()]
            subdirs = [name for name in subdirs if name.stem != 'kidney_3_dense']  # Drop corrupt dataset with no images

            case_index = _copy_files_all_subdirs(data_type, extension, file_channel_suffix, path_target_images,
                                                 path_target_labels, subdirs)

            if data_type == 'train':
                _save_dataset_json(path_target=path_target, num_training=case_index, extension=extension)

            args.skip_preprocess = False  # Rerun processing if source data was updated.
            print(f'Done copying files.')

    if not args.skip_preprocess:
        plan_and_preprocess(dataset_id=dataset_id, check_dataset_integrity=True,
                            configurations_to_run=args.configurations)  # Since data is 2d, only this config is available

    preprocessed_dataset_folder = os.path.join(nnUNet_preprocessed, 'Dataset'+dataset_name_id, 'nnUNetPlans_2d')
    unpack_dataset(preprocessed_dataset_folder, unpack_segmentation=True, overwrite_existing=False,
                   num_processes=max(1, round(get_allowed_n_proc_DA() // 2)))
    zipped_files = list(Path(preprocessed_dataset_folder).glob('*.npz'))
    for file in zipped_files:
        file.unlink()


def _copy_files_all_subdirs(data_type, extension, file_channel_suffix, path_target_images, path_target_labels, subdirs):
    case_index = 0
    for subdir in subdirs:
        print(f'Copying files for acquisition: {subdir.stem}')

        path_origin_images = subdir / 'images'
        path_origin_labels = subdir / 'labels'
        files_origin_images = list(path_origin_images.glob('*' + extension))
        if not files_origin_images:
            raise OSError(f'Files not found at origin dir: {path_origin_images}')

        case_index = _copy_files_for_scan(case_index, data_type, extension, file_channel_suffix,
                                          files_origin_images, path_origin_labels, path_target_images,
                                          path_target_labels)
    return case_index


def _copy_files_for_scan(case_index, data_type, extension, file_channel_suffix, files_origin_images, path_origin_labels,
                         path_target_images, path_target_labels):
    num_files_origin = len(files_origin_images)
    for ind, file_origin_image in enumerate(files_origin_images):
        # Get labels from image filenames to ensure matches. Extra labels will be skipped,
        # images without labels will crash (which could be skipped if data is known to have gaps)

        file_name = file_origin_image.name
        file_target_image = path_target_images / ('BVsegm_' + str(case_index).zfill(4) + '_'
                                                  + file_channel_suffix + extension)
        shutil.copy(file_origin_image, file_target_image)

        if data_type == 'train':
            file_origin_label = get_file(path_origin_labels, file_name)
            file_target_label = path_target_labels / ('BVsegm_' + str(case_index).zfill(4) + extension)
            # Shift labels to expected integer range starting at 1
            label_pil = Image.open(file_origin_label)
            label = np.array(label_pil)
            label[label == 255] = 1
            label_pil = Image.fromarray(label)
            label_pil.save(file_target_label)

        if ind % 100 == 0:
            print(f'{ind + 1}/{num_files_origin}: {file_origin_image} to {file_target_image}')
        case_index += 1
    return case_index


def _setup_paths(dataset_name_id):
    path_origin = Path(r"D:\data\blood-vessel-segmentation")
    nnUnet_raw = Path(os.environ['nnUnet_raw'])
    extension = '.tif'
    file_channel_suffix = '0000'
    path_target = nnUnet_raw / ('Dataset' + dataset_name_id)
    return extension, file_channel_suffix, path_origin, path_target


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
    parser.add_argument('--skip_copy', action='store_true',
                        help="Skip downloading and copying data to nnUnet_raw")
    parser.add_argument('--skip_preprocess', action='store_true',
                        help="Skip preprocessing.")
    parser.add_argument('--configurations', required=False, default=['2d', '3d_fullres', '3d_lowres'],
                        nargs='+',
                        help='[OPTIONAL] Configurations for which the preprocessing should be run. Default: 2d 3d_fullres '
                             '3d_lowres. 3d_cascade_fullres does not need to be specified because it uses the data '
                             'from 3d_fullres. Configurations that do not exist for some dataset will be skipped.')
    args = parser.parse_args()

    if args.skip_preprocess:
        args.skip_copy = True

    return args


if __name__ == '__main__':
    args = parse_command_line_args()

    main(args)
