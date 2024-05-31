import argparse
from pathlib import Path
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.LLM_pytorch_lighting_wrapper import model_setup, tokenizer_setup
from panda_utils import set_display_rows_cols, do_train_val_test_split, read_dataframe

import mlflow

set_display_rows_cols()
np.random.seed(123456)
mlflow.pytorch.autolog()
mlflow.set_experiment('Movie Review Sentiment Analysis')


def main(model_name, train_all=False, extra_class_layers=None):
    model_save_dir = Path(r"D:\Models\LLM") / Path(__file__).stem
    model_save_dir.mkdir(exist_ok=True, parents=True)

    with mlflow.start_run() as run:
        print(f"Starting training run: {run.info.run_id}")
        train_dataloader, val_dataloader, _ = _set_up_dataloading(tokenizer_name=model_name)

        model, trainer = model_setup(model_save_dir,
                                     num_classes=1,
                                     model_name=model_name, do_layer_freeze=not train_all,
                                     extra_class_layers=extra_class_layers)

        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        # To see logs, go to the mlruns folder and type: mlflow ui --port 8080 and go to
        # localhost:8080 in the browser


def _set_up_dataloading(tokenizer_name):
    file_path = r"D:\data\SentimentAnalysisOnMovieReviews\train.tsv"
    # This dataset is heavily resampled with each review being split into smaller chunks down to one letter. The
    # smaller chunks seem to inherit the original review, so the target sentiment for "A"  and "A series",
    # "occasionally amuses" and "none of which amounts to much of a story" all map to the label of the combination of
    # these. More intelligent non-random splitting based on sentence may improve the results here.
    df = read_dataframe(file_path)

    # Convert to positive/negative to frame as classification problem
    df.loc[df['Sentiment'] < 3, 'Sentiment'] = 0
    df.loc[df['Sentiment'] >= 3, 'Sentiment'] = 1

    # The actual test file has no sentiments. We're not competing right now, so just split off a subset of train to
    # test generalizability
    # file_path = r"D:\data\SentimentAnalysisOnMovieReviews\test.tsv"
    # df_test = read_dataframe(file_path)
    df_train, df_val, df_test = do_train_val_test_split(df)

    # # Create an instance of a PandasDataset
    # # Register the dataset in mlflow.
    # # TODO: Ridiculously slow! Refactor to only run once and then fetch
    # dataset_train = mlflow.data.from_pandas(
    #     df_train, source=file_path, name="Movie Sentiment", targets="Sentiment")
    # mlflow.log_input(dataset_train, context="training")
    # dataset_val = mlflow.data.from_pandas(
    #     df_val, source=file_path, name="Movie Sentiment", targets="Sentiment")
    # mlflow.log_input(dataset_val, context="validation")
    # dataset_test = mlflow.data.from_pandas(
    #     df_test, source=file_path, name="Movie Sentiment", targets="Sentiment")
    # mlflow.log_input(dataset_test, context="testing")

    train_dataloader, val_dataloader, test_dataloader = data_loading(df_train=df_train, df_val=df_val, df_test=df_test,
                                                                     tokenizer_name=tokenizer_name)
    return train_dataloader, val_dataloader, test_dataloader


def data_loading(df_train, df_val, df_test, tokenizer_name, subsample=None, batch_size=64):
    # subsample = 1000 for debug
    train_dataset = KaggleSentimentDataset(df_train, tokenizer_name=tokenizer_name, subsample=subsample)
    val_dataset = KaggleSentimentDataset(df_val, tokenizer_name=tokenizer_name, subsample=subsample)
    test_dataset = KaggleSentimentDataset(df_test, tokenizer_name=tokenizer_name, subsample=subsample)
    print(f'The train/val/test split is: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}')
    # num_workers=3 is recommended by lightning. Depends on available resources and cpu.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=3, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=3, persistent_workers=True)
    mlflow.log_param('batch_size', batch_size)
    return train_dataloader, val_dataloader, test_dataloader


def one_hot_encode_sentiment(ratings: pd.Series) -> np.ndarray:
    # initialize an empty, all zero array in size (data length, max_val + 1)
    labels = np.zeros((ratings.size, ratings.max() + 1))

    # add ones in indices where we have a value
    labels[np.arange(ratings.size), ratings] = 1
    return labels


class KaggleSentimentDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer_name, subsample: int = None):
        if subsample is None or subsample > len(df):
            subsample = len(df)
        self.df = df[:subsample]

        tokenizer = tokenizer_setup(tokenizer_name)

        self.examples = self.df['Phrase']
        self.encodings = tokenizer(self.df['Phrase'].to_list(), truncation=True, padding=True)
        self.labels = self.df['Sentiment'].values
        # self.labels_class = self.df['Sentiment'].values

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        # item['labels_class'] = self.labels_class[idx]  # Do not convert to tensor, as this is used for val only
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--model_name', default='distilbert-base-uncased')
    # argument_parser.add_argument('--device', default='cuda:0')
    argument_parser.add_argument('--train_all', action='store_true')
    argument_parser.add_argument('--extra_class_layers', nargs='+',
                                 default=None, type=int,
                                 help="Number of extra layers to add to default classifier head, "
                                      "or list of specified numbers of connections")
    args = argument_parser.parse_args()

    if args.extra_class_layers and len(args.extra_class_layers) == 1:
        args.extra_class_layers = args.extra_class_layers[0]

    main(model_name=args.model_name, train_all=args.train_all, extra_class_layers=args.extra_class_layers)
