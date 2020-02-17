import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


def obtain_bdt(to_plot: bool = False):
    np.random.seed(5521)
    X2 = pd.read_csv('cleaned_data_background2.csv')
    X1 = pd.read_csv('b_mc_data2.csv')[: np.int(2 * len(X2))]
    X = pd.concat([X1, X2], axis=0, ignore_index=True)
    y1 = np.ones(len(X1))
    y2 = np.zeros(len(X2))
    y = np.concatenate([y1, y2], axis=None)
    print(X.columns)
    X = X.drop(columns='Unnamed: 0')
    X['label'] = y
    X = X.sample(frac=1).reset_index(drop=True)
    y = X['label']
    X = X.drop(columns='label')
    n_split = np.int(4*len(X) / 5)
    X_train, X_test = X[:n_split], X[n_split:]
    y_train, y_test = y[:n_split], y[n_split:]
    print(len(y_train), len(y_test))

    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm="SAMME", n_estimators=500,
                             learning_rate=0.01)
    scores = cross_val_score(bdt, X, y, cv=5)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    bdt.fit(X_train, y_train)
    # print(bdt.get_params(), bdt.feature_importances_, bdt.estimator_weights_, bdt.classes_)

    if to_plot:
        test_errors = []
        for test_predict in bdt.staged_predict(X_test):
            test_errors.append(1. - accuracy_score(test_predict, y_test))

        n_trees = len(bdt)
        # print([x for x in bdt.decision_function(X_test)])

        plt.plot(range(1, n_trees + 1), test_errors, c='black', linestyle='dashed')
        plt.show()

        plt.figure(figsize=(10, 5))
        bdt_output = bdt.decision_function(X_test)
        plot_range = (bdt_output.min(), bdt_output.max())
        plt.hist(bdt_output[y_test == 0], bins=10, range=plot_range, label='Not event', alpha=0.3)
        plt.hist(bdt_output[y_test == 1], bins=10, range=plot_range, label='Event', alpha=0.3)
        plt.legend()
        plt.xlabel('Score')
        plt.title('Decision Scores')
        plt.tight_layout()
        plt.show()

        feature_importance = bdt.feature_importances_
        sorted_indices = np.argsort(feature_importance)
        pos = np.arange(sorted_indices.shape[0]) + .5
        plt.barh(pos, feature_importance[sorted_indices])
        plt.yticks(pos, X.columns[sorted_indices])
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()
    return bdt


if __name__ == '__main__':
    obtain_bdt(to_plot=True)
