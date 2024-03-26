import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def train_val_semi_supervised(X: np.ndarray, y: np.ndarray, path_csv: Union[str, Path], model_name: str = 'SVM',
                              erase_labels_fraction: float = 0.7,
                              test_size: float = 0.3, iterations: int = 5):
    # Split includes random shuffle made reproducible by setting the state
    X_train_labeled, X_test, y_train_labeled, y_test = train_test_split(
        X, y, test_size=test_size, random_state=1)

    # Split train data in labeled and (simulated) unlabeled
    X_train_labeled, X_unl, y_train_labeled, y_unl = train_test_split(
        X_train_labeled, y_train_labeled, test_size=erase_labels_fraction, random_state=1)
    print(f"The split is: {X_train_labeled.shape} labeled training cases, {X_unl.shape} unlabeled training cases, and "
          f"{X_test.shape} test cases.")
    if model_name.upper() == 'SVM':
        model = svm.SVC(kernel='linear', probability=True, C=1)
    elif model_name.upper() == 'DT':
        model = DecisionTreeClassifier()
    else:
        raise ValueError(f'Unrecognized model name {model_name}')

    clf = model.fit(X_train_labeled, y_train_labeled)
    print(f"The prediction score on labeled data is {round(clf.score(X_test, y_test), 2)}")
    X_unl_pred_prob = clf.predict_proba(X_unl)

    # Which cutoff to use is a hyperparamter. Optimize a value once in a train set, and use as default for similar data
    conf_cutoffs = [round(cutoff, 2) for cutoff in np.arange(0.45, 1, 0.05)]
    acc_test_ignore = round(clf.score(X_test, y_test), 3)
    print(f"The prediction on test cases without using semi-supervised learning is: {acc_test_ignore}")
    df_test_results = pd.DataFrame()
    for cutoff in conf_cutoffs:
        acc = list()
        num_unlabeled = list()
        X_train_labeled_iter = X_train_labeled.copy()
        y_train_labeled_iter = y_train_labeled.copy()
        X_train_unlabeled_iter = X_unl.copy()
        y_train_unlabeled_iter = y_unl.copy()
        X_unl_pred_prob_iter = X_unl_pred_prob.copy()
        for iter in range(iterations):
            # print(f'Iteration {labeling_iter}: cutoff {cutoff}')
            # conf_ind = df["pred_max_prob"] > cutoff
            conf_ind = np.max(X_unl_pred_prob_iter, axis=1) > cutoff
            # TODO: iteratively add only new examples above threshold
            X_train_labeled_iter = np.append(X_train_labeled_iter, X_train_unlabeled_iter[conf_ind, :], axis=0)
            y_train_labeled_iter = np.append(y_train_labeled_iter, y_train_unlabeled_iter[conf_ind])
            X_train_unlabeled_iter = X_train_unlabeled_iter[~conf_ind, :]
            y_train_unlabeled_iter = y_train_unlabeled_iter[~conf_ind]

            print(f"Iteration {iter} with cutoff {cutoff}. Train cases:  "
                  f"{X_train_labeled_iter.shape[0]}, including {X_train_labeled_iter.shape[0] - X_train_labeled.shape[0]}"
                  f" model-labeled cases. There are {X_train_unlabeled_iter.shape[0]} more unlabeled cases.")

            clf = model.fit(X_train_labeled_iter, y_train_labeled_iter)
            acc_iter = round(clf.score(X_test, y_test), 5)
            print(f"The accuracy for this iteration is {acc_iter}")
            acc.append(acc_iter)
            num_unlabeled.append(X_train_unlabeled_iter.shape[0])
            if X_train_unlabeled_iter.shape[0] == 0:
                acc += [acc[-1]] * (iterations - len(acc))  # Accuracies
                num_unlabeled += [num_unlabeled[-1]] * (iterations - len(num_unlabeled))  # Remaining unlabeled cases
                break
            else:
                X_unl_pred_prob_iter = clf.predict_proba(X_train_unlabeled_iter)
        df_test_results[f"Accuracy: {cutoff} cutoff"] = acc
        df_test_results[f"Unlabeled: {cutoff} cutoff"] = num_unlabeled

    # Add row with no semi-supervised relabeling
    df_test_results['Accuracy: No Relabeling'] = [acc_test_ignore] * iterations
    df_test_results['Unlabeled: No Relabeling'] = [X_unl_pred_prob.shape[0]] * iterations
    df_test_results.to_csv(path_csv)
    return clf
