from pathlib import Path
import numpy as np
import mlflow
from initialize_SROIE import perepare_data
from services.llm_seq2seq_lightning import initialize_model, AbstractiveQAFineTuner
from services.training_setup import trainer_setup

model_name = 't5-base'
# model_name = "google/pegasus-xsum"

np.random.seed(123456)
mlflow.pytorch.autolog()
mlflow.set_experiment('Movie Review Sentiment Analysis')


def main():
    finer_tuner = AbstractiveQAFineTuner()

    with mlflow.start_run() as run:
        print(f"Starting training run: {run.info.run_id}")
        dataloader_train, dataloader_val = perepare_data(finer_tuner.tokenizer, finer_tuner.model_name, batch_size=5)
        # example = next(iter(dataset_train))

        model_save_dir = Path(r"D:\Models\LLM") / (Path(__file__).stem.replace('fine_tune', ''))
        trainer = trainer_setup(finer_tuner.model_name, model_save_dir)

        trainer.fit(finer_tuner, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)


if __name__ == '__main__':
    main()
