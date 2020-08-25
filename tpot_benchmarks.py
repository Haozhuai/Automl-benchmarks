from .data_config import load_data, data_config
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.metrics import roc_auc_score

from autogluon.utils.tabular.ml.models.lr.lr_model import LinearModel
from .preprocessing_utils.featureGenerator import AutoMLFeatureGenerator
import numpy as np
from threading import Thread
import time

import pickle
import os
import signal


# shutdown tpot training thread when time limit is reached
def sleep(duration=20):
    time.sleep(duration)
    p = os.getpid()
    print("current port", p)
    os.kill(p, signal.SIGINT)


# remove linearSVM from default configuration of tpot, because linearSVM does not support probability prediction
classifier_config_dict = {

    # Classifiers
    'sklearn.naive_bayes.GaussianNB': {   # 1
    },

    'sklearn.naive_bayes.BernoulliNB': {   #  6*2 = 12
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]  #
    },

    'sklearn.naive_bayes.MultinomialNB': {  # 12
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },

    'sklearn.tree.DecisionTreeClassifier': {
        'criterion': ["gini", "entropy"],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf':  range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.GradientBoostingClassifier': {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05)
    },

    'sklearn.neighbors.KNeighborsClassifier': {
        'n_neighbors': range(1, 101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },

    'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l1", "l2"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [True, False]
    },

    'xgboost.XGBClassifier': {
        'n_estimators': [100],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'nthread': [1]
    },

    'sklearn.linear_model.SGDClassifier': {
        'loss': ['log', 'hinge', 'modified_huber', 'squared_hinge', 'perceptron'],
        'penalty': ['elasticnet'],
        'alpha': [0.0, 0.01, 0.001],
        'learning_rate': ['invscaling', 'constant'],
        'fit_intercept': [True, False],
        'l1_ratio': [0.25, 0.0, 1.0, 0.75, 0.5],
        'eta0': [0.1, 1.0, 0.01],
        'power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
    },

    # Preprocesssors
    'sklearn.preprocessing.Binarizer': {
        'threshold': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.decomposition.FastICA': {
        'tol': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.cluster.FeatureAgglomeration': {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    },

    'sklearn.preprocessing.MaxAbsScaler': {  # []
    },

    'sklearn.preprocessing.MinMaxScaler': { #
    },

    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2', 'max']
    },

    'sklearn.kernel_approximation.Nystroem': {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        'gamma': np.arange(0.0, 1.01, 0.05),
        'n_components': range(1, 11)
    },

    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },

    'sklearn.preprocessing.PolynomialFeatures': {  # x, x**2
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
    },

    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.preprocessing.RobustScaler': {
    },

    'sklearn.preprocessing.StandardScaler': {
    },

    'tpot.builtins.ZeroCount': {
    },

    'tpot.builtins.OneHotEncoder': {
        'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
        'sparse': [False],
        'threshold': [10]
    },

    # Selectors
    'sklearn.feature_selection.SelectFwe': {
        'alpha': np.arange(0, 0.05, 0.001),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },

    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(1, 100),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },

    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },

    'sklearn.feature_selection.RFE': {
        'step': np.arange(0.05, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    },

    'sklearn.feature_selection.SelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    }

}


if __name__ == '__main__':

    hours = 2  # 2 hours for training
    res = []
    for data_name, d in data_config.items():

        # split dataset train/test = 0.7:0.3
        X_train, X_test, y_train, y_test = load_data(data_name, combine_y=False, split_seed=2020, test_size=0.3)

        # general feature generator;
        feature_generator = AutoMLFeatureGenerator()

        print("#"*50, 'training set preprocessing')
        X_train = feature_generator.fit_transform(X_train, drop_duplicates=False)
        print("#" * 50, 'testing set preprocessing')
        X_test = feature_generator.transform(X_test)

        feature_types_metadata = feature_generator.feature_types_metadata

        problem_type = 'binary'
        path = f'LR-{data_name}'
        name = 'Onehot'
        eval_metric = 'roc_auc'
        stopping_metric = 'roc_auc'

        lr = LinearModel(problem_type=problem_type, path=path, name=name, eval_metric=eval_metric,
                         feature_types_metadata=feature_types_metadata)

        hyperparams = lr.params.copy()

        X_train = lr.preprocess(X_train, is_train=True, vect_max_features=hyperparams['vectorizer_dict_size'],
                                model_specific_preprocessing=True)
        X_test = lr.preprocess(X_test, is_train=False, vect_max_features=hyperparams['vectorizer_dict_size'],
                                model_specific_preprocessing=True)

        t = Thread(target=sleep, args=(hours*60*60,))
        t.start()

        # optimize roc_auc metric
        clf = TPOTClassifier(scoring='roc_auc', random_state=0, verbosity=2,
                             config_dict=classifier_config_dict, population_size=20)
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)[:, 1]

        with open(f'tpot_pred/{data_name}.pickle', 'wb') as f:
            pickle.dump(y_pred, f)

        # get prediction
        auc = roc_auc_score(y_test.values, y_pred)
        res[data_name] = auc
        print(data_name, auc)
        print('#' * 100)
    print(res)
    with open(f"tpot_pred/result.pickle", 'wb') as f:
        pickle.dump(res, f)


