from data_config.data_config import load_data, data_config
import os

if __name__ == '__main__':

    # pass google_dataset folder to google cloud storage, then using google automl do training and prediction
    os.makedirs('google_dataset', exist_ok=True)

    index_col = None
    data_name = 'airlines'
    ylabel = 'Delay'
    # for data_name, d in data_config.items():
        # split dataset train/test = 0.7:0.3
    X_train, X_test = load_data(data_name, combine_y=True, split_seed=2020, test_size=0.3)

    X_train.index.name = 'ID'
    X_test.index.name = 'ID'

    os.makedirs(f"google_dataset/{data_name}", exist_ok=True)

    X_train.to_csv(f"google_dataset/{data_name}/train.csv")
    X_test.to_csv(f"google_dataset/{data_name}/test.csv")

