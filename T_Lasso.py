import pandas as pd
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neighbors
import sklearn.tree
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso, LassoLars
from scipy.stats import pearsonr, ttest_ind, levene
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RepeatedKFold, GridSearchCV, \
    RepeatedStratifiedKFold, LeavePOut, permutation_test_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, mean_squared_error, r2_score, \
    classification_report
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import resample
import itertools
import time
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from matplotlib import text
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
import matplotlib.font_manager as font_manager
import numpy as np
import pickle

from scipy.stats import chi2
import joblib
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=Warning)


xlsx1_filepath = r"C:\Users\70883\Desktop\radiomic_analysis\SL_data\non_multi_val1.csv"
xlsx2_filepath = r"C:\Users\70883\Desktop\radiomic_analysis\SL_data\res_multi_val1.csv"

#
data_1 = pd.read_csv(xlsx1_filepath)
data_2 = pd.read_csv(xlsx2_filepath)
rows_1, cols_1 = data_1.shape
rows_2, cols_2 = data_2.shape
print(rows_1, cols_1)
print(rows_2, cols_2)


data = pd.concat([data_1, data_2])
print(len(data))
data = shuffle(data)

x = data[data.columns[2:]]
y = data['label']
print(data.columns)
colNames = x.columns

# t检验
counts = 0
index = []
for colName in data.columns[2:]:
    if levene(data_1[colName], data_2[colName])[1] > 0.05:
        if ttest_ind(data_1[colName], data_2[colName])[1] < 0.05:
            counts += 1
            index.append(colName)
    else:
        if ttest_ind(data_1[colName], data_2[colName], equal_var=False)[1] < 0.05:
            counts += 1
            index.append(colName)
print(len(index))
# #

if 'label' not in index:
    index = ['label'] + index
data_1 = data_1[index]
data_2 = data_2[index]
data = pd.concat([data_1, data_2])
data = shuffle(data)
# print(data)
data.index = range(len(data))
x = data[data.columns[1:]]
y = data['label']
x = x.apply(pd.to_numeric, errors='ignore')
colNames = x.columns
x = x.fillna(0)
x = x.astype(np.float64)
x = StandardScaler().fit_transform(x)
x = pd.DataFrame(x)
x.columns = colNames

# # LassoCV

alphas = np.logspace(-3,1, 100)
model_LassoCV = LassoCV(alphas=alphas, cv=10, max_iter=100000).fit(x, y)


# # 提取相同数量的特征
coef = pd.Series(model_LassoCV.coef_, index=x.columns)
selected_features = coef[coef != 0].index


x_selected = x[selected_features]
print(x_selected)
print(("Lasso picked" + str(sum(coef != 0))) + "variables and eliminated the other" + str(sum(coef == 0)) + "variables")
index = coef[coef != 0].index
x = x[index]
print(coef[coef != 0])

# RFECV
svc = SVC(kernel="linear", probability=True)
rfecv = RFECV(estimator=svc, step=2, cv=StratifiedKFold(2), scoring='roc_auc', verbose=1, n_jobs=1)
rfecv.fit(x_selected, y)


'''MULTI model'''

# # 在后续运行中，直接使用已固定的特征和参数提取相同的特征数量
x_selected_fixed = x[selected_features]
x_rfecv = x_selected_fixed.iloc[:, rfecv.support_]

# 输出结果
print("Selected Features:")
print(selected_features.tolist())
print("Number of Selected Features:", rfecv.n_features_)
print("RFECV特征选择结果——————————————————————————————————————————————————")
print("有效特征个数 : %d" % rfecv.n_features_)
print("全部特征等级 : %s" % list(rfecv.ranking_))
print("Selected features: ", rfecv.support_)
print("Selected feature column index: ", x_rfecv.columns)
plt.figure()
#  选择的特征数量
plt.xlabel("Number of features selected")
# 交叉验证得分
plt.ylabel("Cross validation score")
# 画出各个特征的得分
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# 计算特征权重
feature_weights = pd.Series(index=x_rfecv.columns, data=rfecv.estimator_.coef_.reshape(-1))
print('Feature weights:')
print(feature_weights)

# 绘制特征权重图
plt.figure(figsize=(12, 8), dpi=300)
sns.barplot(x=feature_weights, y=feature_weights.index)
plt.gcf().subplots_adjust(left=0.20, top=0.90, bottom=0.15, right=None)
plt.title('Feature Weights')
plt.xlabel('Weight')
plt.ylabel('Feature')
plt.show()


feature_weights = model_LassoCV.coef_
feature_names =coef[coef != 0]
# Remove features with zero weights
x_values = feature_weights[feature_weights != 0]

# Create an array of indices for the x-axis tick labels
index = np.arange(len(x_values))

# Plot the bar plot
sns.barplot(x=index, y=x_values)
plt.bar(index, x_values, color='lightblue', edgecolor='black', alpha=0.1)

# Set the x-axis tick labels
plt.xticks(index,feature_names, rotation=45, ha='right', va='top')

# Adjust the plot margins
plt.gcf().subplots_adjust(left=0.15, top=0.90, bottom=0.28, right=None)

# Set the x-axis and y-axis labels
plt.xlabel('feature')
plt.ylabel('weight')

# Display the plot
plt.show()





# # 绘制特征相关性热度图

x_rfecv =x.iloc[:, rfecv.support_]
plt.figure(figsize=(14, 12), dpi=300)
# x_rfecv.columns =  x_rfecv
sns.heatmap(x_rfecv.corr(),
            xticklabels=x_rfecv.columns,
            yticklabels=x_rfecv.columns,
            cmap='RdYlGn',
            center=0.5,
            annot=True)
plt.gcf().subplots_adjust(left=0.15, top=0.90, bottom=0.15, right=None)
plt.title('Correlogram of Selected Features', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


x = x_rfecv
# 参数优化
Cs = np.logspace(-1, 1, 10, base=2)
gammas = np.logspace(-4, 1, 50, base=2)
param_grid = dict(C=Cs, gamma=gammas)
# grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, cv=10).fit(x, y)
grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid=param_grid, cv=10).fit(x, y)

# print(grid.best_params_)
C = grid.best_params_['C']
gamma = grid.best_params_['gamma']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=88)
print(x_train.shape)
model_svm = svm.SVC(kernel='linear', C=C, gamma=gamma, probability=True).fit(x_train, y_train)
score_svm = np.mean(cross_val_score(model_svm, x, y, cv=10))
print(score_svm)



def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)

    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all):
    # ax.plot(thresh_group, net_benefit_model, color = 'crimson', label = 'clinical model'+ ' ({:.3f})'.format(dca_auc))
    ax.plot(thresh_group, net_benefit_model, color='crimson', label='radiomic model' + ' ({:.3f})'.format(dca_auc))
    ax.plot(thresh_group, net_benefit_all, color='black', label='Treat all')
    ax.plot((0, 1), (0, 0), color='black', linestyle=':', label='Treat none')

    # Fill，显示出模型较于treat all和treat none好的部分
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    ax.fill_between(thresh_group, y1, y2, color='crimson', alpha=0.2)

    # Figure Configuration， 美化一下细节
    ax.set_xlim(0, 1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)  # adjustify the y axis limitation
    ax.set_xlabel(
        xlabel='Threshold Probability',
        fontdict={'family': 'Times New Roman', 'fontsize': 15}
    )
    ax.set_ylabel(
        ylabel='Net Benefit',
        fontdict={'family': 'Times New Roman', 'fontsize': 15}
    )
    ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc='upper right')

    return ax


if __name__ == '__main__':
    y_pred_score = model_svm.predict_proba(x_test)[:, 1]
    print(y_pred_score)
    y_label = y_test

    thresh_group = np.arange(0, 1, 0.01)
    net_benefit_model = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)
    net_benefit_all = calculate_net_benefit_all(thresh_group, y_label)
    dca_auc = auc(thresh_group, net_benefit_model)

    fig, ax = plt.subplots()
    ax = plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all)
    # fig.savefig('fig1.png', dpi = 300)
    plt.show()
    # Save the results to variables

    results = {
        'y_pred_score_c': y_pred_score,
        'net_benefit_model_c': net_benefit_model,
        'net_benefit_all': net_benefit_all,
        'thresh': thresh_group}
    with open('C:\\Users\\70883\\result\\result.pkl', 'wb') as f:
        pickle.dump(results, f)

# x = X_RFECV
# 随机森林
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=88)
model_RL = RandomForestClassifier(n_estimators=10000).fit(x_train, y_train)
# score_RL = model_RL.score(x_test, y_test)
score_RL = np.mean(cross_val_score(model_RL, x, y, cv=10))
print('score_RL', score_RL)

# svm
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=88)
model_svm = svm.SVC(kernel='rbf', degree=3, gamma='auto', probability=True).fit(x_train, y_train)
# score_svm = model_svm.score(x_test, y_test)
score_svm = np.mean(cross_val_score(model_svm, x, y, cv=10))
print('score_svm', score_svm)

# 参数优化
Cs = np.logspace(-1, 1, 10, base=2)
gammas = np.logspace(-4, 1, 50, base=2)
param_grid = dict(C=Cs, gamma=gammas)
grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, cv=10).fit(x, y)
# grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, cv=10).fit(x, y)
print(grid.best_params_)
C = grid.best_params_['C']
gamma = grid.best_params_['gamma']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=88)
# model_svm = svm.SVC(kernel='rbf', C=C, gamma=gamma, probability=True).fit(x_train, y_train)
model_svm = svm.SVC(kernel='linear', C=C, gamma=gamma, probability=True).fit(x_train, y_train)
# score_svm = model_svm.score(x_test, y_test)
score_svm = np.mean(cross_val_score(model_svm, x, y, cv=10))
print(score_svm)

# 逻辑回归
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=88)
model_LR = sklearn.linear_model.LogisticRegression(random_state=88).fit(x_train, y_train)
# score_LR = model_LR.score(x_test, y_test)
score_LR = np.mean(cross_val_score(model_LR, x, y, cv=10))
print('score_LR', score_LR)

# 决策树
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=88)
model_DT = sklearn.tree.DecisionTreeClassifier(random_state=88).fit(x_train, y_train)
# score_DT = model_DT.score(x_test, y_test)
score_DT = np.mean(cross_val_score(model_DT, x, y, cv=10))
print('score_DT', score_DT)

# K近邻
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=88)
model_KNN = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5).fit(x_train, y_train)
# score_KNN = model_KNN.score(x_test, y_test)
score_KNN = np.mean(cross_val_score(model_KNN, x, y, cv=10))
print('score_KNN', score_KNN)

# GBM
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=88)
model_GBM = sklearn.ensemble.GradientBoostingClassifier(random_state=88).fit(x_train, y_train)
# score_GBM = model_GBM.score(x_test, y_test)
score_GBM = np.mean(cross_val_score(model_GBM, x, y, cv=10))
print('score_GBM', score_GBM)

# # XGBoost
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=88)
model_XGB = xgb.XGBClassifier(random_state=88, eval_metric='error').fit(x_train, y_train)
scores_XGB = cross_val_score(model_XGB, x, y, cv=10, scoring='accuracy')
mean_score_XGB = np.mean(scores_XGB)
print('XGBoost mean accuracy:', mean_score_XGB)




# ROC曲线
y_probs = model_svm.predict_proba(x)
fpr, tpr, tresholds = roc_curve(y, y_probs[:, 0], pos_label=0)
plt.plot(fpr, tpr, marker='o',
         color='lightblue',
         # edgecolor = 'black',
         alpha=0.8)

plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.show()

auc_score = roc_auc_score(y, model_svm.predict(x))
print(auc_score)
y_test_pred = model_svm.predict(x_test)  # y_test 的预测值
print(classification_report(y_test, y_test_pred))

x = np.array(x).astype("float")
y = np.array(y).astype("float")
# 定义SVM分类器
# clf = SVC(kernel='linear', probability=True)
# clf = SVC(kernel='rbf', C=C, gamma=gamma, probability=True)
# 参数优化
#

clf = SVC(kernel='linear', probability=True)
param_grid = dict(C=Cs, gamma=gammas)
Cs = np.logspace(-1, 1, 10, base=2)
gammas = np.logspace(-4, 1, 50, base=2)
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
grid = GridSearchCV(clf, param_grid, cv=cv).fit(x, y)
print(grid.best_params_)
C = grid.best_params_['C']
gamma = grid.best_params_['gamma']



tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(x, y)):
    clf.fit(x[train], y[train])

    # 在这里改训练集和测试集
    y_prob = clf.predict_proba(x[test])
    fpr, tpr, thresholds = roc_curve(y[test], y_prob[:, 1])
    # y_prob = clf.predict_proba(x[test])
    # fpr, tpr, thresholds = roc_curve(y[test], y_prob[:, 1])
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    ax.plot(fpr, tpr, alpha=0.5, label='ROC fold %d (AUC = %0.2f)' % (i + 1, roc_auc))

# 训练集上的AUC
y_train_pred = clf.predict_proba(x_train)[:, 1]
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_pred)
auc_train = auc(fpr_train, tpr_train)
print("Training AUC: {:.3f}".format(auc_train))

# 测试集上的分类报告
y_pred = clf.predict(x_test)
print("Classification report for test set:")
print(classification_report(y_test, y_pred))

# 绘制平均ROC曲线
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

# Compute upper and lower bounds of 95% CI
std_tpr_CI = np.std(tprs, axis=0)
tprs_upper_CI = np.minimum(mean_tpr + 1.96 * std_tpr_CI, 1)
tprs_lower_CI = np.maximum(mean_tpr - 1.96 * std_tpr_CI, 0)

ax.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2,
        alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

# 绘制标准差阴影区域
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.plot([0, 1], [0, 1], 'k--', lw=2)
ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-0.05, 1.05])
ax.set_xlabel('1-Specificity')
ax.set_ylabel('Sensitivity')
# ax.set_title('Training clinical model')
ax.set_title('radiomic-PS model')
ax.legend(loc="lower right")
plt.show()

# Print the AUC and 95% CI
print("Mean AUC: {:.3f}".format(mean_auc))
print("95% CI for AUC: [{:.3f}, {:.3f}]".format(np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)))

# Print the sensitivity (true positive rate) at a specific specificity value
specificity_value = 0.8  # Adjust the desired specificity value
specificity_index = np.argmin(np.abs(mean_fpr - (1 - specificity_value)))
sensitivity_at_specificity = mean_tpr[specificity_index]
sensitivity_ci_at_specificity = (tprs_lower_CI[specificity_index], tprs_upper_CI[specificity_index])
print("Sensitivity at Specificity {:.1f}: {:.3f} ({:.3f}, {:.3f})".format(specificity_value, sensitivity_at_specificity,
                                                                          sensitivity_ci_at_specificity[0],
                                                                          sensitivity_ci_at_specificity[1]))

# 计算混淆矩阵
confusion = confusion_matrix(y_test, y_pred)

# 计算敏感性（SENS）
sensitivity = confusion[1, 1] / (confusion[1, 1] + confusion[1, 0])

# 计算特异性（SPEC）
specificity = confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])

# 计算精确率（Precision）
precision = confusion[1, 1] / (confusion[1, 1] + confusion[0, 1])

# 计算准确率（Accuracy）
accuracy = (confusion[0, 0] + confusion[1, 1]) / np.sum(confusion)

# 打印结果
print("Sensitivity (Recall): {:.3f}".format(sensitivity))
print("Specificity: {:.3f}".format(specificity))
print("Precision: {:.3f}".format(precision))
print("Accuracy: {:.3f}".format(accuracy))

print("mean_tpr", mean_tpr)
print("mean_fpr", mean_fpr)
print("mean_auc", mean_auc)
# np.savetxt('F:/Doctoral project/data_analysis/ROC_analysis/mean_fprP.txt',mean_fpr)
# np.savetxt('F:/Doctoral project/data_analysis/ROC_analysis/mean_tprP.txt',mean_tpr)


# 均方误差

MSEs = model_LassoCV.mse_path_
"""
MSEs_mean,MSES_std = [],[]
for i in range(len(MSEs)):
    MSEs_mean.append(MSEs[i].mean())
    MSES_std.append(MSEs[i].std())
"""
MSEs_mean = np.apply_along_axis(np.mean, 1, MSEs)
MSEs_std = np.apply_along_axis(np.std, 1, MSEs)

plt.figure(dpi=300)  # dpi = 300
plt.errorbar(model_LassoCV.alphas_, MSEs_mean,  # x,y数据，一一对应
             yerr=MSEs_std,  # y误差范围
             fmt='o',  # 数据点标记
             ms=3,  # 数据点大小
             mfc='r',  # 数据点颜色
             mec='r',  # 数据边缘颜色
             ecolor='lightblue',  # 误差棒颜色
             elinewidth=2,  # 误差棒宽
             capsize=4,  # 误差棒边界线长度
             capthick=1)  # 误差棒边界线厚度
plt.semilogx()
plt.axvline(model_LassoCV.alpha_, color='black', ls='--')
# plt.gcf().subplots_adjust(left=0.25,top=0.95,bottom=0.3, right=None)
plt.xlabel('Lambda')
plt.ylabel('MSE')
ax = plt.gca()
y_major_locator = MultipleLocator(0.05)
ax.yaxis.set_major_locator(y_major_locator)
plt.show()

# 辐射特征的系数
coefs = model_LassoCV.path(x, y, alphas=saved_alphas, max_iter=10000)[1].T
plt.figure(dpi=300)
plt.semilogx(model_LassoCV.alphas_, coefs, '-')
plt.axvline(model_LassoCV.alpha_, color='black', ls='--')
plt.xlabel('Lambda')
plt.ylabel('Coefficients')
plt.show()

