from pathlib import Path

import pandas as pd
from PIL import Image

import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import mlflow
import torch
from torch.utils.data import DataLoader

from ocr_lightning_wrapper import ocr

from utils.os_utils import get_file
from utils.LLM_pytorch_lighting_wrapper import trainer_setup

from transformers import TrOCRProcessor
from utils.ocr_lightning_wrapper import FineTuneTrOCR

mlflow.pytorch.autolog()
mlflow.set_experiment('OCR Receipts')


def main():
    with mlflow.start_run() as run:
        path_top = Path(r"D:\data\CV\SROIE")

        file_annotat = r"D:\data\CV\SROIE\task1_bb_ocr_train"
        model_name = 'microsoft/trocr-large-printed'

        data = dict()
        label_paths = list(path_top.glob('*.txt'))
        for label in label_paths:
            name = str(label).split('.')[0]
            data[name] = dict()
            data[name]['label_path'] = label
            data[name]['image_path'] = get_file(path_top, mask=str(label).split('.')[0] + '.*jpg')

        df = pd.DataFrame.from_dict(data)
        df_train, df_val = train_test_split(df, test_size=0.3)
        processor = TrOCRProcessor.from_pretrained(model_name)

        dataset_train = ImageDatasetJPG(df_train, processor)
        dataset_val = ImageDatasetJPG(df_val, processor)

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


class ImageDatasetJPG(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, processor, subsample: int = None, max_target_length: int = 20):
        if subsample is None or subsample > len(df):
            subsample = len(df)
        self.df = df[:subsample]

        self.processor = processor
        self.max_target_length = 20

    def __getitem__(self, idx):
        # Show - Image.open(self.image_paths[idx]).show()
        image = Image.open(self.df[idx]['image_path']).convert('RGB')
        image = torch.squeeze(self.processor(image, return_tensors="pt").pixel_values).to('cuda')
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        with open(self.df[idx]["label_path"], 'r') as f:
            all_annot = f.readlines()

        bounding_boxes = list()
        text_in_boxes = list()
        for line in all_annot:
            line_split = line.split(',')
            bounding_boxes.append(line_split[0:-1])
            text_in_boxes.append(line_split[-1])

        labels = self.processor.tokenizer(text_in_boxes,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

    def __len__(self):
        return len(self.df)


if __name__ == '__main__':
    main()
