from pytorch_tabnet.tab_model import TabNetClassifier
import pickle
import torch

from sklearn.model_selection import StratifiedKFold
from data_config.data_config import data_config, load_data
import numpy as np
from sklearn.metrics import roc_auc_score


if __name__ == '__main__':

    res = {}
    for data_name in data_config.keys():

        epoch = 100

        train, test, y_train, y_test = load_data(data_name, combine_y=False)

        types = train.dtypes
        #
        categorical_columns = []
        categorical_dims = {}

        features = list(train.columns)

        print(train.shape)

        for col in train.columns:
            if types[col] == 'object':
                train[col] = train[col].fillna("VV_likely")
                test[col] = test[col].fillna('VV_likely')
                d = train[col].unique()
                di = dict(zip(d, range(0, len(d))))
                print(col, len(d))
                train[col] = train[col].map(di).astype(int)
                test.loc[~test[col].isin(d), col] = 'VV_likely'
                test[col] = test[col].map(di).astype(int)
                categorical_columns.append(col)
                categorical_dims[col] = len(d)
            else:
                train.fillna(train[col].mean(), inplace=True)
                test.fillna(test[col].mean(), inplace=True)

        cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]

        cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

        X_train = train.values
        X_test = test.values
        y_train = y_train.values
        y_test = y_test.values

        # max_epochs = 200 if not os.getenv("CI", False) else 2

        skf = StratifiedKFold(shuffle=True, random_state=0)

        test_pred = np.zeros(shape=len(y_test))

        for train_index, val_index in skf.split(X_train, y_train):

            X_tr = X_train[train_index]
            y_tr = y_train[train_index]
            X_val = X_train[val_index]
            y_val = y_train[val_index]

            clf = TabNetClassifier(cat_idxs=cat_idxs,
                                   cat_dims=cat_dims,
                                   cat_emb_dim=1,
                                   optimizer_fn=torch.optim.Adam,
                                   optimizer_params=dict(lr=2e-2),
                                   verbose=0,
                                   scheduler_params={"step_size": 50,  # how to use learning rate scheduler
                                                     "gamma": 0.9},
                                   scheduler_fn=torch.optim.lr_scheduler.StepLR,
                                   mask_type='entmax'  # "sparsemax"
                                   )

            clf.fit(
                X_train=X_tr, y_train=y_tr,
                X_valid=X_val, y_valid=y_val,
                max_epochs=epoch, patience=epoch,
                batch_size=1024, virtual_batch_size=128,
                num_workers=0,
                weights=1,
                drop_last=False
            )

            test_pred += clf.predict_proba(X_test)[:, 1]

        test_pred /= 5

        test_auc = roc_auc_score(y_test, test_pred)

        print('test auc:', test_auc)
        res[data_name] = test_auc

    print(res)
    with open("tabnet_result.pickle", "wb") as f:
        pickle.dump(res, f)







