import cv2
import numpy as np
import torch
from src.models import CRNN
from src.utils import StrLabelConverter, RegionConverter
from src.dataset.utils import preprocess
import os
import pandas as pd
from src.augmentation import *
import yaml
import string

def get_prob(raw_labels, raw_probs, seq_size):
    current_prob = 1.0
    for j in range(seq_size):
        if raw_labels[j] != 0 and not (j > 0 and raw_labels[j] == raw_labels[j - 1]):
            current_prob *= raw_probs[j]

    return current_prob

def get_labels(img_list):
    labels = []
    for path in img_list:
        txt_path = path.replace('.png', '.txt').replace('.jpeg', '.txt').replace('.jpg', '.txt')
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                label = [i.split(',') for i in f.read().strip()]
                labels.append((path, *label))
    return labels

def infer(path, transforms, model, converter, image_w, image_h, use_gpu=True):
    image = cv2.imread(path)

    preprocessed_image = preprocess(image, transform=transforms, image_w=image_w, image_h=image_h).unsqueeze(0)
    if use_gpu:
        image = preprocessed_image.cuda()
    predictions, cls = model(image)
    predictions = predictions.permute(1, 0, 2).contiguous()
    prediction_size = torch.IntTensor([predictions.size(0)]).repeat(1)
    predicted_probs, predicted_labels = predictions.detach().cpu().max(2)
    pred_labels = predicted_labels.permute(1, 0).numpy()
    predicted_probs = torch.exp(predicted_probs).permute(1, 0).numpy()
    probs = get_prob(pred_labels[0], predicted_probs[0], len(converter.alphabet))
    predicted_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=False))

    if cls is not None:
        cls = int(cls.argmax(1))
    return str(predicted_test_labels) + '_' +  str(probs) + '_' + str(cls)

def main(config):
    os.makedirs(config['output_folder'], exist_ok=True)
    img = config['images']
    converter = StrLabelConverter(eval(config['alphabet']))
    region_converter = None
    if config['model']['classification_regions'] is not None:
        region_converter = RegionConverter(config['model']['classification_regions'])
    evaluate = config['evaluate']
    if os.path.isdir(img):
        images = os.listdir(img)
        if evaluate:
            images = get_labels(images)
              
    elif img.endswith('.csv'):
        images = pd.read_csv(img)
    else:
        images = [img]
        if evaluate:
            images = get_labels(images)
    df_images = pd.DataFrame(images)
    if df_images.shape[1] == 1:
        df_images.columns = ['path']
    elif df_images.shape[1] == 3:
        df_images.columns = ['path', 'label', 'region']
    elif df_images.shape[1] == 2:
        df_images.columns = ['path', 'label']
    else:
        raise NotImplementedError
    model_cfg = config['model']
    path_model = model_cfg.pop('path', None)
    model = CRNN(**model_cfg)
    if os.path.isdir(path_model):
        l = [(float(i.split('_')[0].replace(',', '')), i) for i in os.listdir(path_model)]
        path_model = os.path.join(path_model, max(l, key = lambda x: x[0])[1])
    gpu = config['use_gpu']
    state = torch.load(path_model)
    state_dict = state['state_dict']
    new_st = {}
    for k, v in state_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        new_st[k] = v
    model.load_state_dict(new_st)
    if gpu:
        model.cuda()
    model.eval()
    transforms = eval(config['transforms'])
    image_w, image_h = config['image_w'], config['image_h']
    df_images['pred_label_conf_cls'] = df_images.path.apply(infer, args=(transforms, model, converter, image_w, image_h, gpu))
    df_images['conf'] = df_images.pred_label_conf_cls.str.split('_').str[1].astype(np.float32)
    df_images['pred_label'] = df_images.pred_label_conf_cls.str.split('_').str[0]
    df_images['region_pred'] = df_images.pred_label_conf_cls.str.split('_').str[2].astype(int)
    del df_images['pred_label_conf_cls']
    df_images['region_pred_decode'] = df_images.region_pred.apply(region_converter.decode)
    df_images.to_csv(os.path.join(config['output_folder'], 'inference.csv'), index=False)
    with open(os.path.join(config['output_folder'], 'info.txt'), 'w') as f:
        f.write(str(config))
    if evaluate:
        acc_rate = len(df_images[(df_images.conf > config['threshold']) & (df_images.label == df_images.pred_label)])/len(df_images)
        cls_acc_rate = len(df_images[(df_images.region == df_images.region_pred_decode)])/len(df_images)
        with open(os.path.join(config['output_folder'], 'info.txt'), 'a') as f:
            f.write("\nacc_rate: "+str(acc_rate))
            f.write("\ncls_acc_rate: "+str(cls_acc_rate))

if __name__ == '__main__':
    import yaml

    with open('test_config.yml', 'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg)