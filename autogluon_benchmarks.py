"""
wide and deep test, follow code from autogluon
autogluon's NN architecture is based on wide and deep network
"""
from autogluon import TabularPrediction as task
from data_config.data_config import load_data, data_config


if __name__ == '__main__':
    res = {}
    for data_name in data_config.keys():
        ylabel = data_config[data_name]['ylabel']

        X_train, X_valid = load_data(data_name,  combine_y=True)
        train_data = task.Dataset(df=X_train)
        test_data = task.Dataset(df=X_valid)
        savedir = f'{data_name}/'  # where to save trained models
        predictor = task.fit(train_data=train_data,
                             label=ylabel,
                             output_directory=savedir,
                             eval_metric='roc_auc',
                             verbosity=2,
                             visualizer='tensorboard',
                             random_seed=0,
                             save_space=True,
                             keep_only_best=True,
                             )
        auc = predictor.evaluate(X_valid)
        res[data_name] = auc

    print(res)
    import pickle
    with open('autogluon_result.pickle', 'wb') as f:
        pickle.dump(res, f)





