import logging
import os
from torch import nn
import torch

logging.basicConfig(format='[%(asctime)s %(levelname)s:%(name)s:] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('train')


def get_latest_checkpoint(model_dir):
    files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    epoch_numbers = [int(f.split('_')[0]) for f in files]

    max_epoch = max(epoch_numbers)
    max_epoch_index = epoch_numbers.index(max_epoch)
    max_epoch_filename = os.path.join(model_dir, files[max_epoch_index])
    stop = 1
    return max_epoch_filename

def load_weights(model, model_directory, checkpoint_path=None, gpu=0):
    if gpu == 0 and not os.path.exists(model_directory):
        os.makedirs(model_directory, exist_ok=True)

    start_epoch = 0
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{gpu}")
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        start_epoch = checkpoint['epoch']

    elif os.listdir(model_directory):
        checkpoint_path = get_latest_checkpoint(model_directory)
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{gpu}")
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    print(f'Loading from the checkpoint with epoch number {start_epoch}')

    return model, start_epoch + 1
