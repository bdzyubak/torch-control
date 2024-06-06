import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from os_utils import get_matched_files

questions = {'t5-base': {"total": "What is the total amount spent?",
                         "company": "At what company was the purchase made?"}}


def locate_text_inputs():
    path_text_inputs = Path(r"D:\data\CV\SROIE\task1_bb_ocr_train")
    path_labels = Path(r'D:\data\CV\SROIE\task2_summarization_train')

    text_paths, label_paths = get_matched_files(path_text_inputs, path_labels)

    data = {text_path.stem: {'text_path': text_path, 'label_path': label_path} for text_path, label_path in
            zip(text_paths, label_paths)}

    return data


def get_force_max_input_length(model_name, data, max_length=0):
    # Set max_length if data doesn't fit in memory to drop large examples
    longest_question = ""
    for model_name in questions.keys():
        for question_type in questions[model_name]:
            if len(questions[model_name][question_type]) > len(longest_question):
                longest_question = questions[model_name][question_type]

    if max_length == 0:
        for entry in data:
            path = data[entry]['text_path']
            with open(path, 'r') as f:
                text = f.read()

            input_text = make_prompt(model_name, context=text, question=longest_question)
            if len(input_text) > max_length:
                max_length = max_length

    else:
        drop_entries = list()
        for entry in data:
            path = data[entry]['text_path']
            with open(path, 'r') as f:
                text = f.read()

            input_text = make_prompt(model_name, context=text, question=longest_question)

            if len(input_text) > max_length:
                drop_entries.append(entry)
        data_len_orig = len(data)
        data = {key: data[key] for key in data if key not in drop_entries}
        if len(data) < data_len_orig:
            print(f"NOTE: Dropped {data_len_orig - len(data)} entries due to input+question being above "
                  f"specified max length of {max_length}")

    return data, max_length


def make_prompt(model_name, context, question):
    if model_name.startswith('t5'):
        input_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    else:
        raise NotImplementedError(f'The model name {model_name} is not supported.')
    return input_text


class SROIEDatasetTextToLabel(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, model_name: str, tokenizer, subsample: int = None,
                 max_input_length: int = 4000):
        # TODO: Automate setting max target length by searching inputs

        if subsample is None or subsample > len(df):
            subsample = len(df)
        self.df = df[:subsample]

        self.model_name = model_name
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length

    def __getitem__(self, idx):
        with open(self.df.iloc[idx]["text_path"], 'r') as f:
            text_input = f.read()

        prompt_total = make_prompt(model_name=self.model_name, context=text_input,
                                   question=questions[self.model_name]['total'])
        tokens_prompt_total = self.tokenizer(prompt_total,
                                             padding="max_length",
                                             max_length=self.max_input_length)

        prompt_company = make_prompt(model_name=self.model_name, context=text_input,
                                     question=questions[self.model_name]['company'])
        tokens_prompt_company = self.tokenizer(prompt_company,
                                               padding="max_length",
                                               max_length=self.max_input_length)

        with open(self.df.iloc[idx]["label_path"], 'r') as f:
            all_annot = f.read()

        labels_dict = json.loads(all_annot)
        # tokens_company = self.tokenizer(labels_dict['company'],
        #                                 padding="max_length",
        #                                 max_length=25).input_ids
        # tokens_total = self.tokenizer(labels_dict['total'],
        #                               padding="max_length",
        #                               max_length=10).input_ids

        # date_tokens = self.tokenizer.tokenizer(labels_dict['date'],
        #                                           padding="max_length",
        #                                           max_length=self.max_target_length).input_ids
        # address_tokens = self.tokenizer.tokenizer(labels_dict['address'],
        #                                           padding="max_length",
        #                                           max_length=self.max_target_length).input_ids

        encoding = {"text_input": text_input,
                    "prompt_total": [tokens_prompt_total.input_ids, tokens_prompt_total.attention_mask],
                    "prompt_company": [tokens_prompt_company.input_ids, tokens_prompt_company.attention_mask],
                    "company": labels_dict['company'],
                    "total": labels_dict['total']}
        return encoding

    def __len__(self):
        return len(self.df)


def perepare_data(tokenizer, model_name, batch_size=20):
    data = locate_text_inputs()
    max_token_len_prompt = 3000

    data, max_length = get_force_max_input_length(model_name, data, max_length=max_token_len_prompt)

    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.reset_index()
    df_train, df_val = train_test_split(df, test_size=0.3)
    dataset_train = SROIEDatasetTextToLabel(df_train, model_name=model_name, tokenizer=tokenizer,
                                            max_input_length=max_token_len_prompt)
    dataset_val = SROIEDatasetTextToLabel(df_val, model_name=model_name, tokenizer=tokenizer,
                                          max_input_length=max_token_len_prompt)

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
