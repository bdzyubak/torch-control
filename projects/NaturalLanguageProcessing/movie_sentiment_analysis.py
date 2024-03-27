import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, DistilBertTokenizerFast, DistilBertForSequenceClassification, BertConfig
from transformers import AdamW

from panda_utils import set_display_rows_cols

# Based on https://towardsdatascience.com/three-out-of-the-box-transformer-models-4bc4880bc992
# Refactored, converted to Pytorch
set_display_rows_cols()


def main():
    # This dataset is heavily resampled with each review being split into smaller chunks down to one letter. The
    # smaller chunks seem to inherit the original review, so the target sentiment for "A"  and "A series",
    # "occasionally amuses" and "none of which amounts to much of a story" all map to the label of the combination of
    # these. More intelligent non-random splitting based on sentence may improve the results here.
    file_path = r"D:\data\SentimentAnalysisOnMovieReviews\train.tsv"

    dataset_train = KaggleSentimentDataset(file_path, subsample=1000)
    train_loader = DataLoader(dataset_train, batch_size=10, shuffle=True)

    # configuration settings to be used when initializing the model
    config = BertConfig(num_labels=5)  # here we are setting 5 output categories

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    train_model(model, train_loader)


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
    def __init__(self, file_path, subsample: int = None, tok: str = None):
        if file_path.endswith('.csv'):
            self.df = pd.read_csv(file_path)
        elif file_path.endswith('.tsv'):
            self.df = pd.read_csv(file_path, sep='\t')
        else:
            raise ValueError(f'Unsupported file type: {file_path}')

        if subsample is not None:
            self.df = self.df[:subsample]
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

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.labels)


def train_model(model, loader_train, epochs=3, device='cuda:0'):
    model.to(device)
    optim = AdamW(model.parameters(), lr=5e-5)
    for epoch in range(epochs):
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
            if b_ix % 5 == 0:  # 200 train items, 20 batches of 10
                print(" batch = %5d curr batch loss = %0.4f " % \
                      (b_ix, loss.item()))
            print("end epoch = %4d  epoch loss = %0.4f " % \
                  (epoch, epoch_loss))

    print("Training done ")


def predict(model, token_ids, attention_mask):
    y = model.predict(
        {'input_ids': token_ids, 'attention_mask': attention_mask}
    )

    # and we can convert from one-hot encodings to 0 -> 4 ratings with argmax
    preds = np.argmax(y, axis=1)
    return preds


if __name__ == "__main__":
    main()
