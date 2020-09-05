from data_config.data_config import load_data
from sklearn.metrics import roc_auc_score
# specify data
data_name = 'adult-census'

# load data
X_train, X_test, y_train, y_test = load_data(data_name, combine_y=False, split_seed=2020, test_size=0.3)

###################### LR ##########################
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict_proba(X_test)[:, 1]

lr_auc = roc_auc_score(y_test, lr_pred)
print("lr auc:", lr_auc)


###################### KNN ###############################
from sklearn.neighbors import KNeighborsClassifier

# specify neighbors, p, distance metrics
knn = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')
knn.fit(X_train, y_train)
knn_pred = knn.predict_proba(X_test)[:, 1]

knn_auc = roc_auc_score(y_test, knn_pred)
print("knn auc:", knn_auc)

###################### Decision Tree ###############################
from sklearn.tree import DecisionTreeClassifier

# specify split creterion, max depth of tree
dtree = DecisionTreeClassifier(max_depth=6, criterion="gini",)
dtree.fit(X_train, y_train)
dtree_pred = dtree.predict_proba(X_test)[:, 1]

dtree_auc = roc_auc_score(y_test, dtree_pred)
print("decision tree auc:", dtree_auc)

###################### Naive Bayes ###############################
from sklearn.naive_bayes import GaussianNB

# specify neighbors, p, distance metrics
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = dtree.predict_proba(X_test)[:, 1]

nb_auc = roc_auc_score(y_test, nb_pred)
print("naive bayes auc:", nb_auc)

###################### SVM ###############################
from sklearn.svm import SVC

# specify penalty C, and kernel method
svm = SVC(C=1.0, kernel='rbf')
svm.fit(X_train, y_train)
svm_pred = svm.predict_proba(X_test)[:, 1]

svm_auc = roc_auc_score(y_test, svm_pred)
print("support vector machine auc:", svm_auc)

###################### Random Forest ###############################
from sklearn.ensemble import RandomForestClassifier

# specify number of tree,
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_pred = rf.predict_proba(X_test)[:, 1]

rf_auc = roc_auc_score(y_test, rf_pred)
print("random forest auc:", rf_auc)

###################### Extra Tree ###############################
from sklearn.ensemble import ExtraTreesClassifier

# Extra Tree is almost the same as random forest, except it doesnt' do bootstrap
etree = ExtraTreesClassifier(n_estimators=100)
etree.fit(X_train, y_train)
etree_pred = etree.predict_proba(X_test)[:, 1]

etree_auc = roc_auc_score(y_test, etree_pred)
print("extra trees auc:", etree_auc)

###################### Adaboost ###############################
from sklearn.ensemble import AdaBoostClassifier

# boosting algorithms
adaboost = AdaBoostClassifier(n_estimators=100)
adaboost.fit(X_train, y_train)
adaboost_pred = adaboost.predict_proba(X_test)[:, 1]

adaboost_auc = roc_auc_score(y_test, adaboost_pred)
print("adaboost auc:", adaboost_auc)

###################### GBDT ###############################
from sklearn.ensemble import GradientBoostingClassifier

# gbdt uses one-step taylor explason to approximate y
gbdt = GradientBoostingClassifier(n_estimators=100)
gbdt.fit(X_train, y_train)
gbdt_pred = gbdt.predict_proba(X_test)[:, 1]

gbdt_auc = roc_auc_score(y_test, gbdt_pred)
print("gbdt auc:", gbdt_auc)

###################### xgboost ###############################
from xgboost.sklearn import XGBClassifier

# xgboost uses two-step taylor explason to approximate y
xgb = XGBClassifier(n_estimators=100)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict_proba(X_test)[:, 1]

xgb_auc = roc_auc_score(y_test, xgb_pred)
print("xgboost auc:", xgb_auc)

###################### lightgbm ###############################
from lightgbm.sklearn import LGBMClassifier

# lightgbm is much faster and relatively better than xgboost
lgb = LGBMClassifier(n_estimators=100)
lgb.fit(X_train, y_train)
lgb_pred = lgb.predict_proba(X_test)[:, 1]

lgb_auc = roc_auc_score(y_test, lgb_pred)
print("lightgbm auc:", lgb_auc)

###################### catboost ###############################
from catboost import CatBoostClassifier

# catboost is slower, but get better results
cat = CatBoostClassifier(n_estimators=100)
cat.fit(X_train, y_train)
cat_pred = cat.predict_proba(X_test)[:, 1]

cat_auc = roc_auc_score(y_test, cat_pred)
print("catboost auc:", cat_auc)
