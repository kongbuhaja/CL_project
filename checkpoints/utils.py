import os

def dir_check(dataset_name, model_name):
    dataset_dir = f'./checkpoints/{dataset_name}'
    model_dir = f'{dataset_dir}/{model_name}'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        os.makedirs(model_dir)
    elif not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return f'model_dir/{model_dir}'