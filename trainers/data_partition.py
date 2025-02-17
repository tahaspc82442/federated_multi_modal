# partition_data.py (optional helper file)

import random

def partition_dataset_iid(dataset, num_clients=3):
    """
    Example: An IID partition of dataset.train_x
    Keep val/test the same for each client (or you can also split them).
    Return: list of (train_i, val_i, test_i) for each client
    """
    full_train = dataset.train_x[:]  # copy the list
    random.shuffle(full_train)

    train_len = len(full_train)
    chunk_size = train_len // num_clients

    results = []
    for i in range(num_clients):
        start = i * chunk_size
        end = (i+1)*chunk_size if i < num_clients - 1 else train_len
        train_i = full_train[start:end]
        # Let's keep val and test the same for all clients, or you can also split them
        val_i = dataset.val
        test_i = dataset.test
        results.append((train_i, val_i, test_i))
    return results
