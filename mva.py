import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import pickle


def get_x_and_y():
    # x1, x2 = pd.read_csv('b_mc_data2.csv'), pd.read_csv('cleaned_data_background_from_7000mev_protonpid25.csv')
    # x1, x2 = pd.read_csv('B_MC_mass_all_cleaning.csv'), pd.read_csv('cleaned_data_40percentsignal.csv')
    x1, x2 = pd.read_csv('B_MC_mass_taurestriction_nokmu.csv'), pd.read_csv('cleaned_data_2sigma_nokmu.csv')
    x = pd.concat([x1, x2], axis=0, ignore_index=True)
    y1, y2 = np.ones(len(x1)), np.zeros(len(x2))
    y = np.concatenate([y1, y2], axis=None)
    x = x.drop(columns='Unnamed: 0')
    x['label'] = y
    x = x.sample(frac=1).reset_index(drop=True)
    y = x['label']
    x = x.drop(columns='label')
    return x, y


def train_bdt(to_plot=False, save=False):
    np.random.seed(5521)
    x, y = get_x_and_y()
    n_split = np.int(4 * len(x) / 5)
    x_train, x_test = x[:n_split], x[n_split:]
    y_train, y_test = y[:n_split], y[n_split:]
    print(len(y_train), len(y_test))

    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm="SAMME", n_estimators=500,
                             learning_rate=0.01)
    full_sample_weight = np.zeros_like(y)
    full_sample_weight[y == 0] = 1 / (y == 0).sum()
    full_sample_weight[y == 1] = 1 / (y == 1).sum()
    scores = cross_val_score(bdt, x, y, cv=6, fit_params={'sample_weight': full_sample_weight})
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    sample_weight = np.zeros_like(y_train)
    sample_weight[y_train == 0] = 1 / (y_train == 0).sum()
    sample_weight[y_train == 1] = 1 / (y_train == 1).sum()
    bdt.fit(x_train, y_train, sample_weight=sample_weight)

    # test_sample_weight = np.zeros_like(y_test)
    # test_sample_weight[y_test == 0] = 1 / (y_test == 0).sum()
    # test_sample_weight[y_test == 1] = 1 / (y_test == 1).sum()
    print(confusion_matrix(y_test, bdt.predict(x_test), labels=[0, 1]))

    if save:
        pickle.dump(bdt, open('bdt_tau_restriction_no_kmu_2sigma.sav', 'wb'))

    if to_plot:
        test_errors = []
        for test_predict in bdt.staged_predict(x_test):
            test_errors.append(1. - accuracy_score(test_predict, y_test))
        n_trees = len(bdt)

        plt.plot(range(1, n_trees + 1), test_errors, c='black', linestyle='dashed')
        plt.show()

        plt.figure(figsize=(10, 5))
        bdt_output = bdt.decision_function(x_test)
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
        plt.yticks(pos, x.columns[sorted_indices])
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()
    return bdt


def plot_bdt_results():
    X, y = get_x_and_y()
    n_split = np.int(4 * len(X) / 5)
    X_train, X_test = X[:n_split], X[n_split:]
    y_train, y_test = y[:n_split], y[n_split:]
    print(len(y_train), len(y_test))
    bdt = obtain_bdt()
    test_errors = []
    for test_predict in bdt.staged_predict(X_test):
        test_errors.append(1. - accuracy_score(test_predict, y_test))

    n_trees = len(bdt)
    # print([x for x in bdt.decision_function(X_test)])

    plt.plot(range(1, n_trees + 1), test_errors, c='black', linestyle='dashed')
    plt.show()

    plt.figure(figsize=(10, 5))
    bdt_output = bdt.decision_function(X)
    plot_range = (bdt_output.min(), bdt_output.max())
    plt.hist(bdt_output[y == 0], bins=50, range=plot_range, label='Not event', alpha=0.3)
    plt.hist(bdt_output[y == 1], bins=50, range=plot_range, label='Event', alpha=0.3, density=0.0023072060793360356)
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


def obtain_bdt():
    # bdt = pickle.load(open('bdt_normalised_40percentsignal.sav', 'rb'))
    bdt = pickle.load(open('bdt_tau_restriction_no_kmu_2sigma.sav', 'rb'))
    return bdt


if __name__ == '__main__':
    # plot_bdt_results()
    train_bdt(to_plot=True, save=False)
