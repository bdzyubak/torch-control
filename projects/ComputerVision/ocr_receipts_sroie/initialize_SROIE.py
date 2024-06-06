import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from os_utils import get_matched_files


def locate_text_inputs():
    path_text_inputs = Path(r"D:\data\CV\SROIE\task1_bb_ocr_train")
    path_labels = Path(r'D:\data\CV\SROIE\task2_summarization_train')

    text_paths, label_paths = get_matched_files(path_text_inputs, path_labels)

    data = {text_path.stem: {'text_path': text_path, 'label_path': label_path} for text_path, label_path in
            zip(text_paths, label_paths)}

    return data


def make_prompt(model_name, context, question):
    if model_name.startswith('t5'):
        input_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    else:
        raise NotImplementedError(f'The model name {model_name} is not supported.')
    return input_text


class SROIEDatasetTextToLabel(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, model_name: str, tokenizer, subsample: int = None,
                 max_target_length: int = 4000):
        # TODO: Automate setting max target length by searching inputs

        if subsample is None or subsample > len(df):
            subsample = len(df)
        self.df = df[:subsample]

        self.model_name = model_name
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

    def __getitem__(self, idx):
        with open(self.df.iloc[idx]["text_path"], 'r') as f:
            text_input = f.read()

        prompt_total = make_prompt(model_name=self.model_name, context=text_input,
                                   question="What is the total amount spent?")
        tokens_prompt_total = self.tokenizer(prompt_total,
                                             padding="max_length",
                                             max_length=self.max_target_length).input_ids

        prompt_company = make_prompt(model_name=self.model_name, context=text_input,
                                     question="At what company was the purchase made?")
        tokens_prompt_company = self.tokenizer(prompt_company,
                                               padding="max_length",
                                               max_length=self.max_target_length).input_ids

        with open(self.df.iloc[idx]["label_path"], 'r') as f:
            all_annot = f.read()

        labels_dict = json.loads(all_annot)
        tokens_company = self.tokenizer(labels_dict['company'],
                                        padding="max_length",
                                        max_length=25).input_ids
        # date_tokens = self.tokenizer.tokenizer(labels_dict['date'],
        #                                           padding="max_length",
        #                                           max_length=self.max_target_length).input_ids
        # address_tokens = self.tokenizer.tokenizer(labels_dict['address'],
        #                                           padding="max_length",
        #                                           max_length=self.max_target_length).input_ids
        tokens_total = self.tokenizer(labels_dict['total'],
                                      padding="max_length",
                                      max_length=10).input_ids

        encoding = {"text_input": text_input,
                    "prompt_total": torch.tensor(tokens_prompt_total),
                    "prompt_company": torch.tensor(tokens_prompt_company),
                    "company": torch.tensor(tokens_company),
                    "total": torch.tensor(tokens_total)}
        return encoding

    def __len__(self):
        return len(self.df)


def perepare_data(tokenizer, model_name, batch_size=20):
    data = locate_text_inputs()
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.reset_index()
    df_train, df_val = train_test_split(df, test_size=0.3)
    dataset_train = SROIEDatasetTextToLabel(df_train, model_name=model_name, tokenizer=tokenizer)
    dataset_val = SROIEDatasetTextToLabel(df_val, model_name=model_name, tokenizer=tokenizer)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size)
    return dataloader_train, dataloader_val


def get_answer(model_name, tokenizer, model, question, context):
    # Model name is necessary because the suggested prompt format varies
    input_text = make_prompt(model_name=model_name, context=context, question=question)

    features = tokenizer([input_text], return_tensors='pt')
    out = model.generate(input_ids=features['input_ids'].to('cuda'),
                         attention_mask=features['attention_mask'].to('cuda'))
    return tokenizer.decode(out[0])
