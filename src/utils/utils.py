import logging
import os
from torch import nn
import torch
try:
    import src.config.base_config as config
    import src.config.local_config as local_config
    from src.dataset.lp_dataset import LPDataset, LPRegionDataset
    from src.models import base_model as model
    # from src.models import crrn_lite as model
except:
    import config.base_config as config
    import config.local_config as local_config
    from dataset.lp_dataset import LPDataset
    from models import base_model as model
logging.basicConfig(format='[%(asctime)s %(levelname)s:%(name)s:] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('train')


def get_dataset(data_dir, type_key):
    params = local_config.params[type_key]
    params['data_dir'] = data_dir
    dataset = LPDataset(**params)
    return dataset


def get_latest_checkpoint(model_dir):
    files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    epoch_numbers = [int(f.split('_')[0]) for f in files]

    max_epoch = max(epoch_numbers)
    max_epoch_index = epoch_numbers.index(max_epoch)
    max_epoch_filename = os.path.join(model_dir, files[max_epoch_index])
    stop = 1
    return max_epoch_filename

def get_model(config, gpu=0):
    # net = model.CRNN(image_h=config.img_h, num_class=config.num_class, num_layers=config.model_lstm_layers,
    #                  is_lstm_bidirectional=config.model_lsrm_is_bidirectional).cuda(gpu)

    # net = model.CRNN_2(image_h=config.img_h, num_class=config.num_class, num_layers=config.model_lstm_layers,
    #                    is_lstm_bidirectional=config.model_lsrm_is_bidirectional, num_regions=config.num_regions).cuda(gpu)

    # net = model.CRNN_lite(imgH=config.img_h, nc=1, nclass=config.num_class, nh=512).cuda(gpu)

    # net = model.CRNN_resnet(image_h=64, image_w=160, num_class=config.num_class, hl=128, is_lstm_bidirectional=True, linear_size=512).cuda(gpu)

    net = model.CRNN_3(img_channel=3, img_height=config.img_h, img_width=config.img_w, num_class=config.num_class, map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False).cuda(gpu)

    # net = model.CRNN_3_cls(img_channel=3, img_height=config.img_h, img_width=config.img_w, num_class=config.num_class, map_to_seq_hidden=256, rnn_hidden=256, leaky_relu=False, num_regions=config.num_regions).cuda(gpu)

    net = nn.parallel.DistributedDataParallel(net, device_ids=[gpu])
    return net


def load_weights(model, model_directory, checkpoint_path, gpu=0):
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


def save_model(model_dir, model, epoch):
    torch.save(model, os.path.join(model_dir, f'epoch_{epoch}' + config.checkpoint_ext))
