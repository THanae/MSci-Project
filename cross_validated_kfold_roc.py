from scipy import interp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_roc_curve, roc_curve, auc
from sklearn.model_selection import StratifiedKFold

from mva import get_x_and_y

# np.random.seed(5521)

X, y = get_x_and_y()
# background = len(pd.read_csv('cleaned_data_background2.csv'))
# background = len(pd.read_csv('cleaned_data_background_from_7000mev.csv'))
# background = len(pd.read_csv('cleaned_data_background_from_7000mev_protonpid25.csv'))
# background = len(pd.read_csv('cleaned_data_background_from_3sigma_protonpid25.csv'))
# background = len(pd.read_csv('cleaned_data_background_from_7000mev_protonpid25_morefeatures.csv'))
# signal = len(pd.read_csv('b_mc_data2.csv')) * 0.0023072060793360356
# signal = len(pd.read_csv('b_mc_data2.csv')) * 0.0008931248183174746
# signal = len(pd.read_csv('b_mc_data2.csv')) * 0.0023072060793360356
# signal = 120
# background = 2458 - 120
signal = 3
background = 106
print(signal, background)
cv = StratifiedKFold(n_splits=3)
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm="SAMME", n_estimators=500, learning_rate=0.01)
tprs, aucs = [], []
mean_fpr = np.linspace(0, 1, 200000)
mean_tpr = 0

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
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

    sample_weight = np.zeros_like(y[train])
    sample_weight[y[train] == 0] = 1 / (y[train] == 0).sum()
    sample_weight[y[train] == 1] = 1 / (y[train] == 1).sum()
    bdt.fit(X.loc[train], y[train], sample_weight=sample_weight)
    prob = bdt.predict_proba(X.loc[test])
    sample_weight = np.zeros_like(y[test])
    sample_weight[y[test] == 0] = 1 / (y[test] == 0).sum()
    sample_weight[y[test] == 1] = 1 / (y[test] == 1).sum()
    fpr, tpr, thresholds = roc_curve(y[test], prob[:, 1], sample_weight=sample_weight)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    print(mean_tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    ax1.plot(fpr, thresholds, label='threshold fold %d' % i)
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
ax1.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f$\pm$%0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper, tprs_lower = np.minimum(mean_tpr + std_tpr, 1), np.maximum(mean_tpr - std_tpr, 0)
ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

# ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic curve")
ax1.legend(loc="lower right")
sig = signal * np.array(mean_tpr) / np.sqrt(signal * np.array(mean_tpr) + background * np.array(mean_fpr))
e_t = signal * np.array(mean_tpr) / (4 * 10 ** 15 * 280 * 10 ** (-6) * 13.5 / 8)
print(e_t)
sig2 = np.array(mean_tpr) / (5 / 2 + np.sqrt(background * np.array(mean_fpr)))
print(sig)
ax1.axvline(mean_fpr[np.nanargmax(sig)])
ax1.axvline(mean_fpr[np.nanargmax(sig2)])
ax2.plot(mean_fpr, sig, color='k', label=r'Signal significance', lw=2, alpha=.8)
ax2.axvline(mean_fpr[np.nanargmax(sig)])
ax3 = ax2.twinx()
ax3.plot(mean_fpr, sig2, color='b', label=r'Signal significance 2', lw=2, alpha=.8)
ax2.legend(loc="lower right")
plt.show()
