import lib
from autogluon.utils.tabular.ml.models.lr.lr_model import LinearModel
from preprocessing_utils.featureGenerator import AutoMLFeatureGenerator
import numpy as np
from data_config.data_config import load_data, data_config
from sklearn.model_selection import StratifiedKFold
import torch, torch.nn as nn
import torch.nn.functional as F
import pickle
from qhoptim.pyt import QHAdam
from lib.utils import check_numpy, process_in_chunks
from lib.nn_utils import to_one_hot
from sklearn.metrics import roc_auc_score


def predict_logits(self, X_test, device, batch_size=512):
    X_test = torch.as_tensor(X_test, device=device)
    self.model.train(False)
    with torch.no_grad():
        logits = F.softmax(process_in_chunks(self.model, X_test, batch_size=batch_size), dim=1)
        logits = check_numpy(logits)
    return logits


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

        y_train = y_train.values
        y_test = y_test.values

        # X_train = X_train.toarray()
        # X_test = X_test.toarray()

        skf = StratifiedKFold(shuffle=True, random_state=0)

        test_pred = np.zeros(shape=len(y_test))

        test_logits = None

        for train_index, val_index in skf.split(X_train, y_train):
            X_tr = X_train[train_index]
            y_tr = y_train[train_index]
            X_val = X_train[val_index]
            y_val = y_train[val_index]

            data = lib.Dataset(dataset=None, random_state=0,  X_train=X_tr, y_train=y_tr, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test)

            num_features = data.X_train.shape[1]
            num_classes = len(set(data.y_train))

            model = nn.Sequential(
                lib.DenseBlock(num_features, layer_dim=216, num_layers=1, tree_dim=num_classes + 1,
                               flatten_output=False,
                               depth=6, choice_function=lib.entmax15, bin_function=lib.entmoid15),
                lib.Lambda(lambda x: x[..., :num_classes].mean(dim=-2)),
            ).to(device)

            model = model.float()

            with torch.no_grad():
                res = model(torch.as_tensor(data.X_train[:1000], device=device).float())
                # trigger data-aware init

            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)

            optimizer_params = {'nus': (0.7, 1.0), 'betas': (0.95, 0.998)}

            trainer = lib.Trainer(
                model=model, loss_function=F.cross_entropy,
                experiment_name=data_name + '5',
                warm_start=False,
                Optimizer=QHAdam,
                optimizer_params=optimizer_params,
                verbose=True,
                n_last_checkpoints=5
            )

            loss_history, err_history = [], []
            best_val_err = 1.0
            best_step = 0
            early_stopping_rounds = 10_000
            report_frequency = 100

            for batch in lib.iterate_minibatches(data.X_train, data.y_train, batch_size=512,
                                                 shuffle=True, epochs=float('inf')):
                metrics = trainer.train_on_batch(*batch, device=device)

                loss_history.append(metrics['loss'])

                if trainer.step % report_frequency == 0:
                    trainer.save_checkpoint()
                    trainer.average_checkpoints(out_tag='avg')
                    trainer.load_checkpoint(tag='avg')
                    err = trainer.evaluate_classification_error(
                        data.X_valid, data.y_valid, device=device, batch_size=1024)

                    if err < best_val_err:
                        best_val_err = err
                        best_step = trainer.step
                        trainer.save_checkpoint(tag='best')

                    err_history.append(err)
                    trainer.load_checkpoint()  # last
                    trainer.remove_old_temp_checkpoints()

                    # clear_output(True)
                    # plt.figure(figsize=[12, 6])
                    # plt.subplot(1, 2, 1)
                    # plt.plot(loss_history)
                    # plt.grid()
                    # plt.subplot(1, 2, 2)
                    # plt.plot(err_history)
                    # plt.grid()
                    # plt.show()
                    print("Loss %.5f" % (metrics['loss']))
                    print("Val Error Rate: %0.5f" % (err))

                if trainer.step > best_step + early_stopping_rounds:
                    print('BREAK. There is no improvment for {} steps'.format(early_stopping_rounds))
                    print("Best step: ", best_step)
                    print("Best Val Error Rate: %0.5f" % (best_val_err))
                    break
            trainer.load_checkpoint(tag='best')
            logits = predict_logits(data.X_test,  device=device, batch_size=1024)
            if test_logits is None:
                test_logits = logits
            else:
                test_logits += logits
        test_logits /= 5
        auc = roc_auc_score(check_numpy(to_one_hot(y_test)), test_logits)
        print(data_name, auc)
        res[data_name] = auc

    print(res)
    with open("node.pickle", "wb") as f:
        pickle.dump(res, f)











