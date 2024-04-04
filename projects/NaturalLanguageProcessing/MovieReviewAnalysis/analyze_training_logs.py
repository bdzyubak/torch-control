from pathlib import Path

from plotting import plot_tensorboard_logs


plot_tensorboard_logs(model_dir=Path(r"D:\Models\LLM\movie_sentiment_analysis"), model_version="version_2",
                      title='Kaggle Movie Sentiment, DistilBERT Fine Tuning')


