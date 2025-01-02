import os
import copy
import albumentations as A
import cv2
import numpy as np
import torch
from torch import nn
import glob
from models.base_model import CRNN_2
from config import base_config as config
from utils.converter import StrLabelConverter
from dataset.utils import preprocess
import shutil

content_folder = '..'

if __name__ == '__main__':
    model = CRNN_2(image_h=config.img_h, num_class=config.num_class, num_layers=config.model_lstm_layers,
                   is_lstm_bidirectional=config.model_lsrm_is_bidirectional, num_regions=config.num_regions).cuda()
    converter = StrLabelConverter(config.alphabet)
    state = torch.load('../weights/mena_recognizer_iter5_auto.pth')

    state_dict = state['state_dict']
    new_state_dict = copy.deepcopy(state_dict)

    for key in state_dict:
        new_state_dict[key.replace('module.', '')] = new_state_dict.pop(key)

    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()
    regions = config.regions
    transformer = A.Compose([A.NoOp()])

    images = sorted(glob.glob(f'{content_folder}/debug/in/*'))
    out_folder = os.path.join(f'{content_folder}/debug/exp')

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    for idx, image in enumerate(images):
        img = cv2.imread(image)
        if img  is None:
            continue
        preprocessed_image = preprocess(img, transform=transformer).unsqueeze(0)

        cuda_image = preprocessed_image.cuda()

        predictions, cls = model(cuda_image)
        cls_idx = cls.argmax(1).item()
        predictions = predictions.permute(1, 0, 2).contiguous()
        prediction_size = torch.IntTensor([predictions.size(0)]).repeat(1)
        predicted_probs, predicted_labels = predictions.detach().cpu().max(2)
        predicted_probs = np.around(torch.exp(predicted_probs).permute(1, 0).numpy(), decimals=1)
        predicted_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=False))
        predicted_raw_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=True))

        new_path = os.path.join(out_folder, predicted_test_labels.item()+'_'+regions[cls_idx]+'.jpg')
        shutil.copy(image, new_path)
        print(f"path for new plate:{new_path}")
