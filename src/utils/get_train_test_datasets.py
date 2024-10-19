import numpy as np
import datetime

def get_train_test_datasets(data,   # np.array
                            dates,  # list
                            type='random_split',  # 'random' or 'test_after_2021' or 'test_after_covid'
                            seed=0,
                            train_ratio=0.8):  
    
    if type == 'random_split':
        # Generate a random permutation of indices
        num_samples = data.shape[0]
        np.random.seed(seed)
        indices = np.random.permutation(num_samples)

        # Determine the split index
        split_index = int(train_ratio * num_samples)

        # Split the indices for training and testing sets
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]

        # Sort the indices for training and testing sets
        train_indices.sort()
        test_indices.sort()
    elif type == 'temporal_split':
        train_indices = [i for i, val in enumerate(dates) if val.year < 2021]
        test_indices =  [i for i, val in enumerate(dates) if val.year > 2021]      

    # Use the indices to create the train dataset
    data_train = data[train_indices]
    dates_train = np.array(dates)[train_indices]

    # Use the indices to create the test dataset
    data_test = data[test_indices]
    dates_test = np.array(dates)[test_indices]

    return data_train, dates_train, data_test, dates_test  # 4 numpy arrays