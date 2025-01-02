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

content_folder = '/home/user/recognizer_pipeline'
test_set_path = '/home/user/mnt/data/uae/plates/test.csv'
data_dir = '/home/user/mnt'
out_folder = '/home/user/mnt/logs'
df = pd.read_csv(test_set_path)

df['full_path'] = df['image_path'].apply(lambda x: os.path.join(data_dir, x))
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

for idx, (_,label, _, path) in df.iterrows():
    img = cv2.imread(path)
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
    predicted_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=False))
    predicted_raw_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=True))
    stop = 1
    if label != predicted_test_labels.item():
        new_path = os.path.join(out_folder, f"{idx}_{label}_"+predicted_test_labels.item() + '_' + regions[cls_idx] + '.jpg')
        shutil.copy(path, new_path)
        print(idx, path)