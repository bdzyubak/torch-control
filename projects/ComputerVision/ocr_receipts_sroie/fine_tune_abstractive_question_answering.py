from pathlib import Path

from initialize_SROIE import perepare_data
from services.llm_seq2seq_lightning import initialize_model, AbstractiveQAFineTuner
from services.training_setup import trainer_setup

model_name = 't5-base'
# model_name = "google/pegasus-xsum"


def main():
    finer_tuner = AbstractiveQAFineTuner()

    dataloader_train, dataloader_val = perepare_data(finer_tuner.tokenizer, finer_tuner.model_name)
    # example = next(iter(dataset_train))

    model_save_dir = Path(r"D:\Models\LLM") / (Path(__file__).stem.replace('fine_tune', ''))
    trainer = trainer_setup(finer_tuner.model_name, model_save_dir)

    trainer.fit(finer_tuner, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)


if __name__ == '__main__':
    main()