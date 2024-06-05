from pathlib import Path

import pandas as pd
from PIL import Image

import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import mlflow
import torch
from torch.utils.data import DataLoader

from ocr_lightning_wrapper import ocr
from utils.LLM_pytorch_lighting_wrapper import trainer_setup

from transformers import TrOCRProcessor
from utils.ocr_lightning_wrapper import FineTuneTrOCR

mlflow.pytorch.autolog()
mlflow.set_experiment('OCR Receipts')


def main():
    with mlflow.start_run() as run:
        path_top = Path(r"D:\data\CV\Kaggle_text_receipts")

        file_annotat = r"D:\data\CV\Kaggle_text_receipts\annotations.xml"
        model_name = 'microsoft/trocr-large-printed'

        df = get_annot(file_annotat)

        df_train, df_val = train_test_split(df, test_size=0.3)
        processor = TrOCRProcessor.from_pretrained(model_name)

        dataset_train = ImageDatasetJPG(df_train, path_top, processor)
        dataset_val = ImageDatasetJPG(df_val, path_top, processor)

        # dataloader_train = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=True)
        # dataloader_val = DataLoader(dataset_val, batch_size=len(dataset_val), shuffle=True)
        dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
        dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True)

        #
        trocr = FineTuneTrOCR(model_name=model_name, processor=processor)
        # # calling the processor is equivalent to calling the feature extractor
        # pixel_values = processor(example['image'], return_tensors="pt").pixel_values
        # print(pixel_values.shape)

        model_save_dir = Path(r"D:\Models\CV") / str(Path(__file__).stem).replace('train_', '')
        model_save_dir.mkdir(exist_ok=True, parents=True)

        print(f"Starting training run: {run.info.run_id}")

        # generated_ids = trocr.model.generate(next(iter(dataloader_train))['image'], max_new_tokens=100,
        #                                      min_new_tokens=100)
        generated_ids = trocr.model.generate(next(iter(dataloader_train))['image'])
        image_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        trainer = trainer_setup('TrOCR', model_save_dir)
        trainer.fit(trocr, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    # generated_ids = trocr.model.generate(pixel_values)
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
    # df = pd.DataFrame.from_dict(data, orient='index')
    # df.sort_index(inplace=True)
    #
    # columns_str = ['image', 'shop', 'items']
    # for column in columns_str:
    #     df[column] = df[column].str.lower()

    return data


class ImageDatasetJPG(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, path_images_top: Path, processor, subsample: int = None):
        if subsample is None or subsample > len(data):
            subsample = len(data)
        self.data = data[:subsample]

        self.image_paths = [path_images_top / receipt['image'] for receipt in self.data]
        images_missing = list()
        for example in self.image_paths:
            if not example.exists():
                images_missing.append(example)
        if images_missing:
            raise OSError(f"Missing {len(images_missing)} images.")

        self.processor = processor

    def __getitem__(self, idx):
        # Show - Image.open(self.image_paths[idx]).show()
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = torch.squeeze(self.processor(image, return_tensors="pt").pixel_values).to('cuda')

        labels_shop = self.processor.tokenizer(self.data[idx]["shop"],
                                                    padding="max_length",
                                                    max_length=20).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels_shop = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels_shop]

        # self.labels_date = self.processor.tokenizer(self.data[idx]["date"],
        #                                             padding="max_length",
        #                                             max_length=20).input_ids
        # self.labels_items = self.processor.tokenizer(self.data[idx]["items"],
        #                                             padding="max_length",
        #                                             max_length=20).input_ids

        item = {'image': image, 'labels_shop': labels_shop}
        return item

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    main()
