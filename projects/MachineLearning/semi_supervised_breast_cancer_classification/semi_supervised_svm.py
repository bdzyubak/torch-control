import pandas as pd
from sklearn import datasets
from pathlib import Path

from panda_utils import pick_columns_trim_name
from plotting import lineplot, plot_decision_tree_architecture
from services.dataframe_analysis.semisupervised_classification import train_val_semi_supervised


run_training = True  # Either run training/evaluation or load results from csv for display
save_dt_plot = True
path_save = Path(r'D:\ValidationResults') / Path(__file__).parent.name

path_save.mkdir(exist_ok=True, parents=True)
paths_csv = dict()
models = ['DT']  # ['SVM', 'DT']
for model in models:
    paths_csv[model] = path_save / f'iterative_labeling_results_{model}.csv'
    if not paths_csv[model].is_file():
        run_training = True

if run_training:
    # This is a numpy array, not pandas dataset. df['data'] contains n examples x m features, df['target']
    # contains n labels
    df_orig = datasets.load_breast_cancer()
    X = df_orig['data']
    y = df_orig['target']

    for model in models:
        clf = train_val_semi_supervised(X, y, path_csv=paths_csv[model], model_name=model, erase_labels_fraction=0.9)
        if model.upper() == 'DT' and save_dt_plot:
            plot_decision_tree_architecture(clf, df_orig['feature_names'], file_path=path_save / 'architecutre_DT.png')

for model in models:
    df_test_results = pd.read_csv(paths_csv[model])

    feature = 'Accuracy'
    df_subset = pick_columns_trim_name(df_test_results, str_pattern=feature)
    save_file = path_save / ('accuracy_' + model)
    lineplot(df_subset,  save_file=save_file, xlabel='Re-labeling Iteration', ylabel=feature)

    feature = 'Unlabeled'
    df_subset = pick_columns_trim_name(df_test_results, str_pattern=feature)
    save_file = path_save / ('number_unabeled_train_cases_' + model)
    lineplot(df_subset, save_file=path_save / 'number_unabeled_train_cases', xlabel='Re-labeling Iteration',
             ylabel='Number ' + feature)
