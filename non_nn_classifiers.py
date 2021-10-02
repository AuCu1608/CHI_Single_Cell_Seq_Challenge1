import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

path = 'scBALFdata/'
df = pd.read_csv(path + "covid-selected-data.csv")


def get_dataset(return_X=True):
    classes = ['Normal', 'Mild', 'Severe']
    y = []
    labels = pd.read_csv(path + "covid-selected-data-labels.csv").set_index('Unnamed: 0').to_numpy()[:, 0]
    for label in labels:
        y.append(classes.index(label))

    if not return_X:
        return np.array(y, dtype=int)

    X = pd.read_csv(path + "covid-selected-data.csv").set_index('Unnamed: 0').to_numpy()

    return X, np.array(y, dtype=int)


def data_split():
    y = get_dataset(return_X=False)
    X = np.zeros(len(y))
    split_indices = []

    seed = 42
    for i in range(5):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + i)

        for train_index, test_index in skf.split(X, y):
            split_indices.append((train_index, test_index))

    return split_indices


# Parameters for different classifiers: LR: {'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True,
# 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 10000, 'multi_class': 'multinomial', 'n_jobs': None,
# 'penalty': 'l2', 'random_state': None, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False} SVM: {
# 'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0, 'decision_function_shape':
# 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'probability': True, 'random_state': None,
# 'shrinking': True, 'tol': 0.001, 'verbose': False} RF: {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None,
# 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None,
# 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2,
# 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': None,
# 'verbose': 0, 'warm_start': False} XGBoost: {'objective': 'multi:softprob', 'use_label_encoder': False,
# 'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 1,
# 'gamma': 0, 'gpu_id': -1, 'importance_type': 'gain', 'interaction_constraints': '', 'learning_rate': 0.300000012,
# 'max_delta_step': 0, 'max_depth': 6, 'min_child_weight': 1, 'missing': nan, 'monotone_constraints': '()',
# 'n_estimators': 100, 'n_jobs': 10, 'num_parallel_tree': 1, 'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1,
# 'scale_pos_weight': None, 'subsample': 1, 'tree_method': 'exact', 'validate_parameters': 1, 'verbosity': None,
# 'eval_metric': 'mlogloss'}


def train_and_test(model_name):
    X, y = get_dataset()
    split_indices = data_split()
    auroc_list = []
    iter = 0

    for train_indices, test_indices in split_indices:
        print('\r Training iter %d' % iter, end='')
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        if model_name == 'LR':
            model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000)
        elif model_name == 'SVM_linear':
            model = SVC(kernel='linear', probability=True)
        elif model_name == 'SVM_rbf':
            model = SVC(kernel='rbf', probability=True)
        elif model_name == 'XGBoost':
            model = XGBClassifier(eval_metric='mlogloss', n_jobs=10, use_label_encoder=False)
        elif model_name == 'RF':
            model = RandomForestClassifier()
        else:
            print('No matched classifiers')
            return

        model.fit(X_train, y_train)

        y_test_pred_proba = model.predict_proba(X_test)
        auroc_ovr_macro = roc_auc_score(y_test, y_test_pred_proba, average='macro', multi_class='ovr')
        auroc_list.append(auroc_ovr_macro)

        iter += 1

    return np.array(auroc_list)


def print_res(auroc_list):
    print('Mean auroc: %.3f%%' % (np.mean(auroc_list)*100))
    print('Auroc std: %.3f%%' % (np.std(auroc_list)*100))


def rank_feature(model_name):
    X, y = get_dataset()

    if model_name == 'XGBoost':
        model = XGBClassifier(eval_metric='mlogloss', n_jobs=10, use_label_encoder=False)
    elif model_name == 'RF':
        model = RandomForestClassifier()

    model.fit(X, y)

    feat_imp = pd.DataFrame()
    feat_imp["feat_name"] = df.columns.to_numpy()[1:]
    feat_imp["gain_importance"] = model.feature_importances_
    feat_imp.sort_values(by=['gain_importance'], ascending=False, inplace=True)
    print(feat_imp)


if __name__ == '__main__':
    train_and_test('LR')
    train_and_test('SVM_linear')
    train_and_test('SVM_rbf')
    train_and_test('XGBoost')
    train_and_test('RF')
    rank_feature('RF')
    rank_feature('XGBoost')