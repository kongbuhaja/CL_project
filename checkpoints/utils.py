import os

def dir_check(dataset_name):
    dataset_dir = f'./checkpoints/{dataset_name}'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
