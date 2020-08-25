import pandas as pd
from sklearn.model_selection import train_test_split


data_config = {'adult-census': {'index_col': 'ID', 'ylabel': 'class'},
              'ailerons': {'index_col': None, 'ylabel': 'binaryClass'},
              'Amazon_employee_access': {'index_col':None, 'ylabel': 'target'},
              "bank-marketing": {'index_col':None, 'ylabel': 'Class'},
              'BayesianNetworkGenerator_vote': {'index_col':None, 'ylabel':'Class'},
              'Click_prediction_small': {'index_col': None, 'ylabel': 'click'},
              'CreditCardSubset': {'index_col': None, 'ylabel': 'Class'},
              'dataset_3_kr-vs-kp': {'index_col': None, 'ylabel': 'class'},
              'eeg-eye-state': {'index_col': None, 'ylabel': 'Class'},
              'elevators': {'index_col': None, 'ylabel': 'binaryClass'},
              'fried': {'index_col': None, 'ylabel': 'binaryClass'},
              'kdd_ipums_la_97-small': {'index_col': None, 'ylabel': 'binaryClass'},
              'letter': {'index_col': None, 'ylabel': 'binaryClass'},
              'MagicTelescope': {'index_col': 'ID', 'ylabel': 'class:'},
              'New_aps_failure': {'index_col': None, 'ylabel': 'class'},
              'New_higgs': {'index_col': None, 'ylabel': 'class'},
              'New_KDDCup09_appetency': {'index_col': None, 'ylabel': 'APPETENCY'},
              'New_KDDCup09_churn': {'index_col': None, 'ylabel': 'CHURN'},
              'New_KDDCup09_upselling': {'index_col': None, 'ylabel': 'UPSELLING'},
              'New_kick': {'index_col': None, 'ylabel': 'IsBadBuy'},
              'New_test_dataset': {'index_col': None, 'ylabel': 'class'},
              'nomao': {'index_col': None, 'ylabel': 'Class'},
              'pendigits': {'index_col': None, 'ylabel': 'binaryClass'},
              'Run_or_walk_information': {'index_col': None, 'ylabel': 'activity'},
              'skin-segmentation': {'index_col': None, 'ylabel': 'Class'},
              'sylva_prior': {'index_col': None, 'ylabel': 'label'}
              }


def load_data(data_name,  combine_y=False, split_seed=2020, test_size=0.3):

    data_path = f'dataset/{data_name}.csv'

    df = pd.read_csv(data_path, index_col=data_config[data_name]["index_col"])
    ylabel = data_config[data_name]['ylabel']

    # if not DataDict[data_name]['label_transform']:
    df[ylabel], _ = df[ylabel].factorize()

    X = df.drop(ylabel, axis=1)

    y = df[ylabel]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=split_seed, stratify=y)

    if combine_y:

        X = pd.concat([X_train, y_train], axis=1)
        X_valid = pd.concat([X_test, y_test], axis=1)
        return X, X_valid
    else:
        return X_train, X_test, y_train, y_test