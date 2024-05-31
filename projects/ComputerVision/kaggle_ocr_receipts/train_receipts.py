from pathlib import Path

from PIL import Image

import pandas as pd
import xml.etree.ElementTree as ET

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel


def main():
    path_top = Path(r"D:\data\CV\Kaggle_text_receipts")

    file_annotat = r"D:\data\CV\Kaggle_text_receipts\annotations.xml"
    df = get_annot(file_annotat)

    # paths_images_df = [name for name in df[:]['image'].values]
    # paths_images_os = path_images_top.glob('*')
    # paths_images_os = [str(name) for name in paths_images_os if str(name).lower().endswith('.jpg')]
    # TODO: Add formatting to enable sanity check
    # images_missing_in_df = [name for name in paths_images_os if name not in paths_images_df]
    # images_missing_in_os = [name for name in paths_images_df if name not in paths_images_os]
    # if images_missing_in_os:
    #     raise OSError(f"{len(images_missing_in_os)} images present in df are missing from drive")
    # elif images_missing_in_df:
    #     print(f'WARNING: {len(images_missing_in_df)} images present on drive are missing from df')

    df_train, df_val = train_test_split(df, test_size=0.3)

    dataset_train = ImageDatasetJPG(df_train, path_top)
    dataset_val = ImageDatasetJPG(df_val, path_top)

    dataloader_train = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=True)
    dataloader_val = DataLoader(dataset_train, batch_size=len(dataset_val), shuffle=True)

    # processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
    # # calling the processor is equivalent to calling the feature extractor
    # pixel_values = processor(image, return_tensors="pt").pixel_values
    # print(pixel_values.shape)
    #
    # model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
    #
    # generated_ids = model.generate(pixel_values)
    # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(generated_text)


def get_annot(file_annotat):
    tree = ET.parse(file_annotat)
    root = tree.getroot()
    data = dict()
    for receipt in root:
        if receipt.tag in ['version', 'meta']:
            continue

        id = int(receipt.attrib['id'])
        data[id] = dict()
        data[id]['image'] = receipt.attrib['name']
        for category in receipt:
            if category.tag == 'box':
                if category.attrib['label'] == 'shop':
                    data[id]['shop'] = category[0].text
                elif category.attrib['label'] == 'date_time':
                    data[id]['date'] = category[0].text
                elif category.attrib['label'] == 'item':
                    if 'items' not in data[id]:
                        data[id]['items'] = [category[0].text]
                    else:
                        data[id]['items'].append(category[0].text)
    df = pd.DataFrame.from_dict(data, orient='index')
    df.sort_index(inplace=True)

    columns_str = ['image', 'shop', 'items']
    for column in columns_str:
        df[column] = df[column].str.lower()

    return df


class ImageDatasetJPG(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, path_images_top: Path, subsample: int = None):
        if subsample is None or subsample > len(df):
            subsample = len(df)
        self.df = df[:subsample]

        self.examples = [path_images_top / name for name in self.df[:]['image'].values]
        images_missing = list()
        for example in self.examples:
            if not example.exists():
                images_missing.append(example)
        if images_missing:
            raise OSError(f"Missing {len(images_missing)} images.")

        label_columns = [name for name in df.columns if name != 'image']
        self.labels = self.df[label_columns]

    def __getitem__(self, idx):
        image = torch.tensor(Image.open(self.examples[idx]), dtype=torch.float32)
        labels = self.df[idx]

        item = {'image': image, 'labels': labels}
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    main()
