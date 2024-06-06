import torch

from projects.ComputerVision.ocr_receipts_sroie.initialize_SROIE import perepare_data, get_answer
from services.llm_seq2seq_lightning import initialize_model

model_name = 't5-base'
# model_name = "google/pegasus-xsum"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = initialize_model(model_name)

    dataset_train = perepare_data(tokenizer)
    example = next(iter(dataset_train))

    get_answer(model_name, model, tokenizer, question='At what company was the purchase made?',
               context=example['text_input'])


if __name__ == '__main__':
    main()
