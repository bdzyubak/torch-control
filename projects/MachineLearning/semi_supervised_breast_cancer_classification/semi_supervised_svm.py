import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams.update({'font.size': 22})  # must set in top
from pathlib import Path


run_training = False  # Either run training/evaluation or load results from csv for display
path_csv = Path(r'D:\ValidationResults') / Path(__file__).parent.name / 'iterative_labelig_results.csv'

if not path_csv.is_file:
    run_training = True

if run_training:
    path_csv.parent.mkdir(exist_ok=True, parents=True)
    df_orig = datasets.load_breast_cancer()
    # This is a numpy array, not pandas dataset. df['data'] contains n examples x m features, df['target'] contains n labels
    X = df_orig['data']
    y = df_orig['target']

    class_names = df_orig['target_names']

    # Split includes random shuffle made reproducible by setting the state
    X_train_labeled, X_test, y_train_labeled, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1)
    # Split train data in labeled and (simulated) unlabeled
    X_train_labeled, X_unl, y_train_labeled, y_unl = train_test_split(
        X_train_labeled, y_train_labeled, test_size=0.7, random_state=1)

    print(f"The split is: {X_train_labeled.shape} labeled training cases, {X_unl.shape} unlabeled training cases, and "
          f"{X_test.shape} test cases.")

    clf = svm.SVC(kernel='linear', probability=True, C=1).fit(X_train_labeled, y_train_labeled)
    print(f"The prediction score on labeled data is {round(clf.score(X_test, y_test), 2)}")

    X_unl_pred_prob = clf.predict_proba(X_unl)
    X_unl_pred_class = clf.predict(X_unl)

    df = pd.DataFrame(X_unl_pred_prob, columns=class_names)
    for class_ind, column in enumerate(class_names):
        df['pred' + column] = X_unl_pred_prob[:, class_ind]

    df['pred_max_prob'] = df[class_names].max(axis=1)  # Prediction confidence
    df['pred_class'] = X_unl_pred_class

    # Which cutoff to use is a hyperparamter. Optimize a value once in a train set, and use as default for similar data
    conf_cutoffs = [round(cutoff, 2) for cutoff in np.arange(0.45, 1, 0.05)]

    iterations = 5
    acc_test_ignore = round(clf.score(X_test, y_test), 3)

    print(f"The prediction on test cases without using semi-supervised learning is: {acc_test_ignore}")
    iteration_names = [f"Iter_{name}_acc" for name in range(1, iterations + 1)]

    df_test_results = pd.DataFrame(columns=['no_relabel']+conf_cutoffs)
    for cutoff in conf_cutoffs:
        acc = list()
        X_train_labeled_iter = X_train_labeled.copy()
        y_train_labeled_iter = y_train_labeled.copy()
        X_train_unlabeled_iter = X_unl.copy()
        y_train_unlabeled_iter = y_unl.copy()
        X_unl_pred_prob_iter = X_unl_pred_prob.copy()
        for iter_name in iteration_names:
            # print(f'Iteration {labeling_iter}: cutoff {cutoff}')
            # conf_ind = df["pred_max_prob"] > cutoff
            conf_ind = np.max(X_unl_pred_prob_iter, axis=1) > cutoff
            # TODO: iteratively add only new examples above threshold
            X_train_labeled_iter = np.append(X_train_labeled_iter, X_train_unlabeled_iter[conf_ind, :], axis=0)
            y_train_labeled_iter = np.append(y_train_labeled_iter, y_train_unlabeled_iter[conf_ind])
            X_train_unlabeled_iter = X_train_unlabeled_iter[~conf_ind, :]
            y_train_unlabeled_iter = y_train_unlabeled_iter[~conf_ind]

            print(f"{iter_name} with cutoff {cutoff}. Train cases:  "
                  f"{X_train_labeled_iter.shape[0]}, including {X_train_labeled_iter.shape[0] - X_train_labeled.shape[0]}"
                  f" model-labeled cases. There are {X_train_unlabeled_iter.shape[0]} more unlabeled cases.")

            clf = svm.SVC(kernel='linear', probability=True).fit(X_train_labeled_iter, y_train_labeled_iter)
            acc_iter = round(clf.score(X_test, y_test), 5)
            print(f"The accuracy for this iteration is {acc_iter}")
            acc.append(acc_iter)
            if X_train_unlabeled_iter.shape[0] == 0:
                acc += [acc[-1]] * (len(iteration_names) - len(acc))
                break
            else:
                X_unl_pred_prob_iter = clf.predict_proba(X_train_unlabeled_iter)
        df_test_results[cutoff] = acc

    # Add row with no semi-supervised relabeling
    df_test_results['no_relabel'] = [acc_test_ignore] * iterations

    df_test_results.to_csv(path_csv)

df_test_results = pd.read_csv(path_csv)


plt.figure()

# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
color_palette = sns.color_palette()
# plot = sns.lineplot(data=df_test_results['no_relabel'], palette=color_palette)
plot = sns.lineplot(data=df_test_results[df_test_results.columns[1:]], palette=color_palette, legend=True)
sns.move_legend(plot.axes, 'center right')
plt.xlabel('Re-labeling Iteration')
plt.ylabel('Classification Accuracy')
plt.show()
