from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import loggers

from transformers import AutoTokenizer
from transformers import AdamW

from services.LLM_pytorch_lighting_wrapper import FineTuneLLM, qc_requested_models_supported
from panda_utils import set_display_rows_cols, do_train_val_test_split
from torch_utils import tensor_to_numpy

# Based on https://towardsdatascience.com/three-out-of-the-box-transformer-models-4bc4880bc992
# Refactored, converted to Pytorch
set_display_rows_cols()
np.random.seed(123456)


def main():
    # This dataset is heavily resampled with each review being split into smaller chunks down to one letter. The
    # smaller chunks seem to inherit the original review, so the target sentiment for "A"  and "A series",
    # "occasionally amuses" and "none of which amounts to much of a story" all map to the label of the combination of
    # these. More intelligent non-random splitting based on sentence may improve the results here.
    file_path = r"D:\data\SentimentAnalysisOnMovieReviews\train.tsv"
    df = read_dataframe(file_path)

    model_names = ['distilbert-base-uncased']
    # model_names = ['distilbert-base-uncased', 'ProsusAI/finbert']
    # qc_requested_models_supported(model_names)

    # The actual test file has no sentiments. We're not competing right now, so just split off a subset of train to
    # test generalizability
    # file_path = r"D:\data\SentimentAnalysisOnMovieReviews\test.tsv"
    # df_test = read_dataframe(file_path)
    df_test, df_train, df_val = do_train_val_test_split(df)

    model_save_dir = Path(r"D:\Models\LLM") / Path(__file__).stem
    model_save_dir.mkdir(exist_ok=True, parents=True)

    train_dataloader, val_dataloader, test_dataloader = data_loading(df_train=df_train, df_val=df_val, df_test=df_test)

    for model_name in model_names:
        model, trainer = model_setup(model_save_dir,
                                     num_classes=train_dataloader.dataset.__getitem__(0)['labels'].shape[0],
                                     model_name=model_name)
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        # To see logs, run in command line: tensorboard --logdir=model_save_dir/[version_run_number] and go to the
        # localhost:6006 in the browser

    predict(model, test_dataloader)


def model_setup(save_dir, num_classes, model_name='distilbert-base-uncased'):
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir,
                                          filename=model_name+"-{epoch:02d}-{val_loss:.2f}",
                                          save_top_k=1,
                                          monitor="val_acc")
    early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.0001, patience=5, verbose=False, mode="max")
    tb_logger = loggers.TensorBoardLogger(save_dir=save_dir)
    model = FineTuneLLM(num_classes=num_classes,
                        model_name=model_name)
    trainer = pl.Trainer(max_epochs=100, callbacks=[checkpoint_callback, early_stop_callback], logger=tb_logger,
                         log_every_n_steps=1)
    return model, trainer


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


def read_dataframe(file_path, nrows=None):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, nrows=nrows)
    elif file_path.endswith('.tsv'):
        df = pd.read_csv(file_path, nrows=nrows, sep='\t')
    else:
        raise ValueError(f'Unsupported file type: {file_path}')
    return df


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


def train_model(model, loader_train, model_save_file=None, loader_val=None, epochs=3, device='cuda:0'):
    model.to(device)
    optim = AdamW(model.parameters(), lr=5e-5)
    for epoch in range(1, epochs):
        model.train()
        epoch_loss = 0.0
        for (b_ix, batch) in enumerate(loader_train):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            epoch_loss += loss.item()  # accumulate batch loss
            loss.backward()
            optim.step()

        print(f"Epoch {epoch} loss: {epoch_loss}")

        if loader_val is not None:
            model.eval()
            preds = predict(model, loader_val, device='cuda:0')
            acc_epoch = accuracy_score(preds, loader_val.dataset.labels_class)
            print(f"Epoch {epoch} val accuracy: {acc_epoch}")

        if model_save_file is not None and epoch % 10 == 0:
            torch.save(model.state_dict(), model_save_file.parent / (model_save_file.stem + f'_epoch{epoch}.pth'))


    print("Training done ")
    model.eval()


def predict(model, dataloader_test, device='cuda:0'):
    model.eval()
    batch_size = dataloader_test.batch_size
    preds = np.zeros([len(dataloader_test)*batch_size])
    for idx, batch in enumerate(dataloader_test):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = model(input_ids, attention_mask=attention_mask)
        preds_batch = torch.argmax(output['logits'], axis=1)
        preds[idx*batch_size:(idx*batch_size+batch_size)] = tensor_to_numpy(preds_batch)

    return preds


if __name__ == "__main__":
    main()
