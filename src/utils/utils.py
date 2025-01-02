import logging
import os

import config.base_config as config
import config.local_config as local_config
import torch
from dataset.lp_dataset import LPDataset
from models import base_model as model

logging.basicConfig(
    format='[%(asctime)s %(levelname)s:%(name)s:] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger('train')


def get_dataset(data_dir, type_key):
    params = local_config.params[type_key]
    params['data_dir'] = data_dir
    dataset = LPDataset(**params)
    return dataset


def get_model(load_weights, model_dir, gpu):
    net = model.CRNN(
        image_h=config.img_h,
        num_class=config.num_class,
        num_layers=config.model_lstm_layers,
        is_lstm_bidirectional=config.model_lsrm_is_bidirectional
    ).cuda(gpu)
    # net = nn.parallel.DistributedDataParallel(net, device_ids=[gpu])

    start_epoch = 0
    if load_weights is not None:
        net.load_state_dict(
            torch.load(os.path.join(model_dir, f'epoch_{load_weights}' + config.checkpoint_ext))['state_dict'],
            strict=False)
        start_epoch = load_weights + 1

    logger.info(f'Loading from the checkpoint with epoch number {start_epoch}')
    return net, start_epoch


def save_model(model_dir, model, epoch):
    torch.save(model, os.path.join(model_dir, f'epoch_{epoch}' + config.checkpoint_ext))
