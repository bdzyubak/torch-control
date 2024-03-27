import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from transformers import AutoTokenizer, DistilBertForSequenceClassification
from transformers import AdamW

from panda_utils import set_display_rows_cols
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

    # The actual test file has no sentiments. We're not competing right now, so just split off a subset of train to
    # test generalizability
    # file_path = r"D:\data\SentimentAnalysisOnMovieReviews\test.tsv"
    # df_test = read_dataframe(file_path)
    df_train, df_test_val = train_test_split(df, train_size=0.6)
    df_val, df_test = train_test_split(df, train_size=0.5)

    model_save_file = Path(r"D:\Models\LLM") / Path(__file__).stem / 'distilbert.pth'
    model_save_file.parent.mkdir(exist_ok=True, parents=True)

    dataset_train = KaggleSentimentDataset(df_train)  # subsample = 1000 for debug
    dataset_val = KaggleSentimentDataset(df_val)
    dataset_test = KaggleSentimentDataset(df_test)

    loader_train = DataLoader(dataset_train, batch_size=10, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=10)
    loader_test = DataLoader(dataset_test, batch_size=10)

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                num_labels=dataset_train.__getitem__(0)['labels'].shape[0])
    train_model(model, loader_train, loader_val=loader_val, epochs=100)

    torch.save(model.state_dict(), model_save_file.parent / (model_save_file.stem + '_final.pth'))

    predict(model, loader_test)


def read_dataframe(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.tsv'):
        df = pd.read_csv(file_path, sep='\t')
    else:
        raise ValueError(f'Unsupported file type: {file_path}')
    return df


# def read_tokenize_one_hot_labels_alternative(file_path, tokenizer):
#     df_train, token_ids_train, attention_mask_train = read_tokenize_data(file_path, tokenizer)
#     labels_train = one_hot_encode_sentiment(df_train['Sentiment'].values)
#     return labels_train
#
#
# def read_tokenize_data(file_path, tokenizer):
#     # initialize lists to contain input IDs and attention masks
#     token_ids = list()
#     attention_mask = list()
#
#     df = pd.read_csv(file_path, sep='\t')
#     df = df[:1000]
#     # loop through each sentence in the input data and encode to ID and mask tensors
#     for sentence in df['Phrase'].tolist():
#         tokens = tokenizer.encode_plus(
#             sentence, max_length=50, truncation=True, padding='max_length',
#             add_special_tokens=True, return_attention_mask=True,
#             return_token_type_ids=False, return_tensors='pt')  # return_tensors=pt - Pytorch
#
#         # add the input ID and attention mask tensors to respective lists
#         token_ids.append(tokens['input_ids'])
#         attention_mask.append(tokens['attention_mask'])
#
#     return df, token_ids, attention_mask


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
        self.labels_one_hot = one_hot_encode_sentiment(self.df['Sentiment'].values)
        self.labels_class = self.df['Sentiment'].values

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels_one_hot[idx], dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.labels_one_hot)


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
