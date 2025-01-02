import os
import copy
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
import glob
from models.base_model import CRNN_2
from config import base_config as config
from utils.converter import StrLabelConverter
from dataset.utils import preprocess
import shutil
from pathlib import Path

model = CRNN_2(image_h=config.img_h, num_class=config.num_class, num_layers=config.model_lstm_layers,
               is_lstm_bidirectional=config.model_lsrm_is_bidirectional, num_regions=config.num_regions).cuda()
converter = StrLabelConverter(config.alphabet)
state = torch.load('../weights/model.pth')

state_dict = state['state_dict']
new_state_dict = copy.deepcopy(state_dict)

for key in state_dict:
    new_state_dict[key.replace('module.', '')] = new_state_dict.pop(key)

model.load_state_dict(new_state_dict)
model.cuda()
model.eval()
regions = ["dubai", "abu-dhabi", "sharjah", "ajman", "ras-al-khaimah", "fujairah", "alquwain"]
transformer = A.Compose([A.NoOp()])

iteration_num = 1
folder = f'/home/user/mnt/data/uae/plates/iteration-{iteration_num}'
error_folder = f'/home/user/mnt/data/uae/plates/errors/iteration-{iteration_num}'
images = glob.glob(os.path.join(folder, "**", "*.jpg"))
for idx, image_path in enumerate(images):
    img = cv2.imread(image_path)
    if img is None or isinstance(img, type(None)):
        print('Image is None')
        continue
    preprocessed_image = preprocess(img, transform=transformer).unsqueeze(0)
    cuda_image = preprocessed_image.cuda()

    predictions, cls = model(cuda_image)
    cls_idx = cls.argmax(1).item()
    predictions = predictions.permute(1, 0, 2).contiguous()
    prediction_size = torch.IntTensor([predictions.size(0)]).repeat(1)
    predicted_probs, predicted_labels = predictions.detach().cpu().max(2)
    predicted_probs = np.around(torch.exp(predicted_probs).permute(1, 0).numpy(), decimals=1)
    predicted_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=False)).item()
    predicted_raw_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=True))

    label_path = image_path.replace(".jpg", ".txt")
    with open(label_path, 'r') as f:
        l = f.read().split(',')
    label = l[0]
    whitelist = set('abcdefghijklmnopqrstuvwxyz1234567890')
    label = str(label).strip().lower().replace('\n', '').replace(' ', '').replace('!', '').replace('#', '').replace('@',
                                                                                                                    '').replace(
        '?', '').replace('$', '').replace('-', '').replace('.', '').replace('|', '').replace('_', '').replace('`',
                                                                                                              '').replace(
        '=', '').encode("ascii", "ignore").decode()

    label = ''.join(filter(whitelist.__contains__, label))
    if label != predicted_test_labels:
        out_folder = Path(os.path.join(error_folder, os.path.basename(os.path.dirname(image_path))))
        if not out_folder.exists():
            out_folder.mkdir()

        new_image_path = os.path.join(out_folder, os.path.basename(image_path))
        shutil.copy(image_path, new_image_path)

        data = f'{image_path.replace("/home/user/mnt/", "")},{label},{predicted_test_labels}'
        with open(new_image_path.replace('.jpg', '.txt'), 'w') as f:
            f.write(data)
        print(f'Working with data:{idx},{image_path}')
