import argparse
from pathlib import Path
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from utils.LLM_pytorch_lighting_wrapper import model_setup
from panda_utils import set_display_rows_cols, do_train_val_test_split, read_dataframe


set_display_rows_cols()
np.random.seed(123456)


def main(model_name, freeze_pretrained_weights=True):
    if freeze_pretrained_weights:
        model_save_file = model_name + '_frozen_last_checkpoint.pth'
    else:
        model_save_file = model_name + '_last_checkpoint.pth'

    train_dataloader, val_dataloader = _set_up_dataloading()

    model_save_dir = Path(r"D:\Models\LLM") / Path(__file__).stem
    model_save_dir.mkdir(exist_ok=True, parents=True)
    # Prefer to get number of classes procedurally, but this requires loading data. For now, hard specify to debug new
    # models. The clean solution is to load just one sample datapoint to get number of labels.
    model, trainer = model_setup(model_save_dir,
                                 num_classes=5,
                                 # num_classes=train_dataloader.dataset.__getitem__(0)['labels'].shape[0],
                                 model_name=model_name)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # To see logs, run in command line: tensorboard --logdir=model_save_dir/[version_run_number] and go to the
    # localhost:6006 in the browser

    torch.save(model.state_dict(), model_save_dir / model_save_file)


def _set_up_dataloading():
    file_path = r"D:\data\SentimentAnalysisOnMovieReviews\train.tsv"
    # This dataset is heavily resampled with each review being split into smaller chunks down to one letter. The
    # smaller chunks seem to inherit the original review, so the target sentiment for "A"  and "A series",
    # "occasionally amuses" and "none of which amounts to much of a story" all map to the label of the combination of
    # these. More intelligent non-random splitting based on sentence may improve the results here.
    df = read_dataframe(file_path)
    # The actual test file has no sentiments. We're not competing right now, so just split off a subset of train to
    # test generalizability
    # file_path = r"D:\data\SentimentAnalysisOnMovieReviews\test.tsv"
    # df_test = read_dataframe(file_path)
    df_test, df_train, df_val = do_train_val_test_split(df)
    train_dataloader, val_dataloader, test_dataloader = data_loading(df_train=df_train, df_val=df_val, df_test=df_test)
    return train_dataloader, val_dataloader


def data_loading(df_train, df_val, df_test, subsample=None):
    train_dataset = KaggleSentimentDataset(df_train, subsample=subsample)  # subsample = 1000 for debug
    val_dataset = KaggleSentimentDataset(df_val, subsample=subsample)
    test_dataset = KaggleSentimentDataset(df_test, subsample=subsample)
    print(f'The train/val/test split is: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}')
    # num_workers=3 is recommended by lightining. Depends on available resources and cpu.
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=3, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=10, num_workers=3, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=10, num_workers=3, persistent_workers=True)
    return train_dataloader, val_dataloader, test_dataloader


def one_hot_encode_sentiment(ratings: pd.Series) -> np.ndarray:
    # initialize an empty, all zero array in size (data length, max_val + 1)
    labels = np.zeros((ratings.size, ratings.max() + 1))

    # add ones in indices where we have a value
    labels[np.arange(ratings.size), ratings] = 1
    return labels


class KaggleSentimentDataset(torch.utils.data.Dataset):
    def __init__(self, df, subsample: int = None, tok: str = None):
        if subsample is None or subsample > len(df):
            subsample = len(df)
        self.df = df[:subsample]

        if tok is None:
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tok)
            except:
                raise ValueError(f'Unsupported tokenizer {tok}')

        self.examples = self.df['Phrase']
        self.encodings = tokenizer(self.df['Phrase'].to_list(), truncation=True, padding=True)
        self.labels = one_hot_encode_sentiment(self.df['Sentiment'].values)
        self.labels_class = self.df['Sentiment'].values

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        item['labels_class'] = self.labels_class[idx]  # Do not convert to tensor, as this is used for val only
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--model_name', default='distilbert-base-uncased')
    # argument_parser.add_argument('--device', default='cuda:0')
    argument_parser.add_argument('-freeze', default=True)
    args = argument_parser.parse_args()

    main(model_name=args.model_name, freeze_pretrained_weights=args.freeze)
