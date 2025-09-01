import numpy as np
import os

import src
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat):
        self.X_num = X_num
        self.X_cat = X_cat

    def __getitem__(self, index):
        this_num = self.X_num[index]
        this_cat = self.X_cat[index]

        sample = (this_num, this_cat)

        return sample

    def __len__(self):
        return self.X_num.shape[0]

def preprocess(dataset_path, task_type='binclass', inverse=False, cat_encoding=None, concat=True):
    
    T_dict = {}

    T_dict['normalization'] = "quantile"
    T_dict['num_nan_policy'] = 'mean'
    T_dict['cat_nan_policy'] = None
    T_dict['cat_min_frequency'] = None
    T_dict['cat_encoding'] = cat_encoding
    T_dict['y_policy'] = "default"

    T = src.Transformations(**T_dict)

    dataset = make_dataset(
        data_path=dataset_path,
        T=T,
        task_type=task_type,
        change_val=False,
        concat=concat
    )
    data = dataset.X_num if dataset.X_num is not None else dataset.X_cat
    N_train, N_test = len(data['train']), len(data['test'])

    if cat_encoding is None:
        X_num = dataset.X_num
        X_cat = dataset.X_cat

        # Initialize placeholders for outputs
        X_train_num, X_test_num = [[]] * N_train, [[]] * N_test
        X_train_cat, X_test_cat = [[]] * N_train, [[]] * N_test
        categories, d_numerical = [], 0

        # Assign values if X_num and X_cat are not None
        if X_num is not None:
            X_train_num, X_test_num = X_num.get('train'), X_num.get('test')
            d_numerical = X_train_num.shape[1] if X_train_num is not None else 0
        X_num = (X_train_num, X_test_num)
            

        if X_cat is not None:
            X_train_cat, X_test_cat = X_cat.get('train'), X_cat.get('test')
            categories = src.get_categories(X_train_cat) if X_train_cat is not None else []
        X_cat = (X_train_cat, X_test_cat)

        # Handle inverse transformation if needed
        if inverse:
            num_inverse = dataset.num_transform.inverse_transform if d_numerical > 0 else None
            cat_inverse = dataset.cat_transform.inverse_transform if len(categories) > 0 else None

            return X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse
        else:
            return X_num, X_cat, categories, d_numerical
    else:
        return dataset


def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for target, source in zip(target_params, source_params):
        target.detach().mul_(rate).add_(source.detach(), alpha=1 - rate)



def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


def make_dataset( 
    data_path: str,
    T: src.Transformations,
    task_type,
    change_val: bool,
    concat = True,
):

    # Helper function to load data and check for empty arrays
    def load_and_check(file_path):
        if os.path.exists(file_path):
            data = np.load(file_path,allow_pickle=True)
            if data.size == 0 or data.shape[0] == 0 or (len(data.shape) > 1 and data.shape[1] == 0):
                return None
            return data
        return None

    # classification
    if task_type == 'binclass' or task_type == 'multiclass':
        X_cat = load_and_check(os.path.join(data_path, 'X_cat_train.npy'))
        X_num = load_and_check(os.path.join(data_path, 'X_num_train.npy'))
        y = load_and_check(os.path.join(data_path, 'y_train.npy'))

        X_cat = {} if X_cat is not None else None
        X_num = {} if X_num is not None else None
        y = {} if y is not None else None

        for split in ['train', 'test']:
            X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                if concat:
                    X_cat_t = concat_y_to_X(X_cat_t, y_t)
                X_cat[split] = X_cat_t  
            if y is not None:
                y[split] = y_t
    else:
        # regression
        X_cat = load_and_check(os.path.join(data_path, 'X_cat_train.npy'))
        X_num = load_and_check(os.path.join(data_path, 'X_num_train.npy'))
        y = load_and_check(os.path.join(data_path, 'y_train.npy'))

        X_cat = {} if X_cat is not None else None
        X_num = {} if X_num is not None else None
        y = {} if y is not None else None

        for split in ['train', 'test']:
            X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)
            if X_num is not None:
                if concat:
                    X_num_t = concat_y_to_X(X_num_t, y_t)
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            if y is not None:
                y[split] = y_t

    info = src.load_json(os.path.join(data_path, 'info.json'))

    D = src.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=src.TaskType(info['task_type']),
        n_classes=info.get('n_classes')
    )

    if change_val:
        D = src.change_val(D)

    return src.transform_dataset(D, T, None)