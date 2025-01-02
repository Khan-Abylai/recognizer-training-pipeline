import copy
import os
import shutil

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch

from config import base_config as config
from dataset.utils import preprocess
from models.base_model import CRNN_2
from utils.converter import StrLabelConverter


def generate_plate_template(real_plate_number):
    template = ""
    for char in real_plate_number:
        if char.isalpha():
            template += 'A'
        elif char.isdigit():
            template += '9'
        else:
            template += char  # Keep any non-alphanumeric characters as they are
    return template


def process(data, engine, regions_list, out_folder, data_dir, list_file_path):
    list_files = []
    for idx, row in data.iterrows():
        file_path = os.path.join(data_dir, row.image_path)
        raw_label = row.car_labels

        whitelist = set('abcdefghijklmnopqrstuvwxyz1234567890')
        label = str(raw_label).strip().lower().replace('\n', '').replace(' ', '').replace('!', '').replace('#',
                                                                                                           '').replace(
            '@', '').replace('?', '').replace('$', '').replace('-', '').replace('.', '').replace('|', '').replace('_',
                                                                                                                  '').replace(
            '`', '').replace('=', '').encode("ascii", "ignore").decode()

        label = ''.join(filter(whitelist.__contains__, label))
        img = cv2.imread(file_path)
        if img is None:
            continue
        preprocessed_image = preprocess(img, transform=transformer).unsqueeze(0)
        cuda_image = preprocessed_image.cuda()

        predictions, cls = engine(cuda_image)
        cls_idx = cls.argmax(1).item()
        predictions = predictions.permute(1, 0, 2).contiguous()
        prediction_size = torch.IntTensor([predictions.size(0)]).repeat(1)
        predicted_probs, predicted_labels = predictions.detach().cpu().max(2)
        predicted_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=False))

        pred_label = predicted_test_labels.item()

        if label != pred_label:
            template = generate_plate_template(pred_label)
            template_folder = os.path.join(out_folder, template)

            if len(pred_label) == 0:
                template_folder = os.path.join(out_folder, "empty")

            if not os.path.exists(template_folder):
                os.makedirs(template_folder, exist_ok=True)

            dst_file_path = os.path.join(template_folder, os.path.basename(file_path))
            shutil.copy(file_path, dst_file_path)
            extension = os.path.basename(file_path).split(".")[-1]
            label_file_path = dst_file_path.replace(f".{extension}", ".txt")
            with open(label_file_path, "w") as f:
                f.write(f"{pred_label},{row.region}")
            list_files.append(row.image_path)
            print(f'{dst_file_path} proceed - real:{label} pred:{pred_label}')
        print(f'proceeding idx:{idx}.')

    df = pd.DataFrame(data=list_files, columns=['image_path'])
    df.to_csv(list_file_path, index=False)


if __name__ == '__main__':
    model = CRNN_2(image_h=config.img_h, num_class=config.num_class, num_layers=config.model_lstm_layers,
                   is_lstm_bidirectional=config.model_lsrm_is_bidirectional, num_regions=config.num_regions).cuda()
    converter = StrLabelConverter(config.alphabet)
    state = torch.load('../weights/recognizer_mena_iter5.pth')

    state_dict = state['state_dict']
    new_state_dict = copy.deepcopy(state_dict)
    for key in state_dict:
        new_state_dict[key.replace('module.', '')] = new_state_dict.pop(key)

    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()
    regions = config.regions
    transformer = A.Compose([A.NoOp()])

    output_folder = '/workspace/data/debug'

    train = pd.read_csv('/workspace/data/uae/uae_iteration_5/train.csv')
    test = pd.read_csv("/workspace/data/uae/uae_iteration_5/test.csv")

    data_folder = '/workspace'
    train_list_file_path = '/workspace/data/debug/train.csv'
    test_list_file_path = '/workspace/data/debug/test.csv'
    process(train, model, regions, output_folder, data_folder, train_list_file_path)
    process(test, model, regions, output_folder, data_folder, test_list_file_path)
