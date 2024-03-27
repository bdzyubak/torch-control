import pandas as pd
from sklearn import datasets

from panda_utils import pick_columns_trim_name
from plotting import lineplot

from pathlib import Path

from services.dataframe_analysis.semisupervised_classification import train_val_semi_supervised


run_training = False  # Either run training/evaluation or load results from csv for display
path_save = Path(r'D:\ValidationResults') / Path(__file__).parent.name
path_csv = path_save / 'iterative_labeling_results.csv'

if not path_csv.is_file:
    run_training = False

if run_training:
    path_save.mkdir(exist_ok=True, parents=True)
    df_orig = datasets.load_breast_cancer()
    # This is a numpy array, not pandas dataset. df['data'] contains n examples x m features, df['target'] contains n labels
    X = df_orig['data']
    y = df_orig['target']

    class_names = df_orig['target_names']

    train_val_semi_supervised(X, y, path_csv=path_csv, model_name='SVM', erase_labels_fraction=0.9)

df_test_results = pd.read_csv(path_csv)

feature = 'Accuracy'
df_subset = pick_columns_trim_name(df_test_results, str_pattern=feature)
lineplot(df_subset,  save_file=path_save / 'accuracy', xlabel='Re-labeling Iteration', ylabel=feature)

feature = 'Unlabeled'
df_subset = pick_columns_trim_name(df_test_results, str_pattern=feature)
lineplot(df_subset, save_file=path_save / 'number_unabeled_train_cases', xlabel='Re-labeling Iteration',
         ylabel='Number ' + feature)
