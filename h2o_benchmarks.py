import h2o, time
from h2o.automl import H2OAutoML
import pandas as pd
import argparse
from data_config.data_config import load_data


def h2o_train(X_train, X_test, y_train, y_test, seed=2020):

    X_train_c = X_train.copy()
    X_test_c = X_test.copy()

    target_name = y_train.name

    cols = list(X_train_c.columns)
    cat_cols = [col for col in X_train_c.columns if X_train_c[col].dtype == 'O']

    Train = h2o.H2OFrame.from_python(pd.concat([X_train_c, y_train], axis=1))

    Train[target_name] = Train[target_name].asfactor()

    for col in cat_cols:
        Train[col] = Train[col].asfactor()

    model = H2OAutoML(seed=seed, max_runtime_secs=3600*2)

    model.train(x=cols, y=target_name, training_frame=Train)

    print('modeling steps: ', model.modeling_steps)
    print('modeling learderboard', model.leaderboard)
    print('modeling log', model.event_log)
    print('modeling leader', model.leader)

    Test = h2o.H2OFrame.from_python(pd.concat([X_test_c, y_test], axis=1))
    Test[target_name] = Test[target_name].asfactor()

    for col in cat_cols:
        Test[col] = Test[col].asfactor()  # encoding would influenced here?
    pred = model.predict(Test).as_data_frame().values[:, 2]

    from sklearn.metrics import roc_auc_score
    h2o_auc = roc_auc_score(y_test, pred)
    h2o.cluster().shutdown()
    print('result auc:', h2o_auc)

    return h2o_auc


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Modeling h2o')
    parser.add_argument('--dataset', '-d', help="pass dataset name", required=True)
    parser.add_argument('--seed', '-s', help='random seed', default=2020)

    args = parser.parse_args()

    data_name = str(args.dataset)
    seed = int(args.seed)

    h2o.init()

    X_train, X_test, y_train, y_test = load_data(data_name)

    start_time = time.time()
    score = h2o_train(X_train, X_test, y_train, y_test, seed=2020)
    end_time = time.time()

    hour = (end_time - start_time) / 3600.0

    with open('h2o_result.txt', 'a') as f:
        f.write(f"{data_name}\t{score}\t{hour}\n")
