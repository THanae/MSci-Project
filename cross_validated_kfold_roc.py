from scipy import interp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm, datasets
from sklearn.metrics import plot_roc_curve, roc_curve, accuracy_score, confusion_matrix, auc
from sklearn.model_selection import StratifiedKFold

from mva import get_x_and_y

X, y = get_x_and_y()
cv = StratifiedKFold(n_splits=3)
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm="SAMME", n_estimators=500, learning_rate=0.01)
tprs, aucs = [], []
mean_fpr = np.linspace(0, 1, 100)
mean_tpr = 0

fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    # print(i)
    # y_train = y[train]
    # x_train = X.loc[train]
    # sample_weight = np.zeros_like(y_train)
    # # sample_weight[y_train == 0] = 1 / (y_train == 0).sum()
    # # sample_weight[y_train == 1] = 1 / (y_train == 1).sum()
    # sample_weight[y_train == 0] = 1
    # sample_weight[y_train == 1] = 0.0023072060793360356
    # bdt.fit(x_train, y_train, sample_weight=sample_weight)
    # sample_weight = np.zeros_like(y[test])
    # sample_weight[y[test] == 0] = 1
    # sample_weight[y[test] == 1] = 0.0023072060793360356
    # viz = plot_roc_curve(bdt, X.loc[test], y[test], name='ROC fold {}'.format(i), alpha=0.3, lw=1, ax=ax, sample_weight=sample_weight)
    # interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
    # interp_tpr[0] = 0.0
    # tprs.append(interp_tpr)
    # aucs.append(viz.roc_auc)

    bdt.fit(X.loc[train], y[train])
    probas_ = bdt.predict_proba(X.loc[test])
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    print(mean_tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    plt.plot(fpr, thresholds, label='threshold fold %d' % i)
    interp_tpr = interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(roc_auc)

# ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
# mean_tpr /= len(cv)
print(mean_tpr)
mean_tpr[-1] = 1.0
mean_auc, std_auc = auc(mean_fpr, mean_tpr), np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f$\pm$%0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper, tprs_lower = np.minimum(mean_tpr + std_tpr, 1), np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic curve")
ax.legend(loc="lower right")
plt.show()

