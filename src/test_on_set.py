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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report


content_folder = '/home/yeleussinova/data_SSD/eu_plate_data'
test_set_path = '/home/yeleussinova/data_SSD/eu_plate_data/test.csv'
data_dir = '/home/yeleussinova/data_SSD/eu_plate_data'
out_folder = '/home/yeleussinova/data_SSD/eu_out'
df = pd.read_csv(test_set_path)
print("test size: ", len(df))

df['full_path'] = df['image_path'].apply(lambda x: os.path.join(data_dir, x))
model = CRNN_2(image_h=config.img_h, num_class=config.num_class, num_layers=config.model_lstm_layers,
                   is_lstm_bidirectional=config.model_lsrm_is_bidirectional, num_regions=config.num_regions).cuda()
converter = StrLabelConverter(config.alphabet)
state = torch.load('../weights/model_eu/eu_final.pth')

state_dict = state['state_dict']
new_state_dict = copy.deepcopy(state_dict)

for key in state_dict:
    new_state_dict[key.replace('module.', '')] = new_state_dict.pop(key)

model.load_state_dict(new_state_dict)
model.cuda()
model.eval()
regions = ['albania', 'andorra', 'austria', 'belgium', 'bosnia', 'bulgaria', 'croatia', 'cyprus', 'czech', 'estonia',
           'finland', 'france', 'germany', 'greece', 'hungary', 'ireland', 'italy', 'latvia',
           'licht', 'lithuania', 'luxemburg', 'makedonia', 'malta', 'monaco', 'montenegro', 'netherlands', 'poland',
           'portugal', 'romania', 'san_marino', 'serbia', 'slovakia', 'slovenia', 'spain', 'sweden', 'swiss']
transformer = A.Compose([A.NoOp()])

correct_pred = {classname: 0 for classname in regions}
total_pred = {classname: 0 for classname in regions}
confusion_matrix = np.zeros((36, 36))

y_test = []
y_pred_list = []
count = 0
eu_acuts = 'åäćčđöüšž'
true_labels = 0
for idx, (_,label, true_region, path) in df.iterrows():
    path = path.replace("europe_last/", '')

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
    label = str(label).strip().lower().replace('\n', '').replace(' ', '').replace('!', '').replace('#', '').replace(
        '@', '').replace('?', '').replace('$', '').replace('-', '').replace('.', '').replace('|', '').replace('_','').replace(
        '`', '').replace('=', '')
    predicted_test_labels = predicted_test_labels.item().replace('@', 'å').replace('&', 'ä').replace('!', 'ć').replace('?', 'č').replace('%', 'đ').replace('^', 'ö').replace('#', 'ü').replace('$', 'š').replace('|', 'ž')

    if label != predicted_test_labels:
        new_path = os.path.join(out_folder, f"{idx}_{label}_"+predicted_test_labels + '_' + regions[cls_idx] + '.jpg')
        shutil.copy(path, new_path)
        # print(idx, path)
    else:
        true_labels += 1
    if regions[cls_idx] == true_region:
        correct_pred[regions[cls_idx]] += 1
    total_pred[regions[cls_idx]] += 1
    confusion_matrix[regions.index(true_region), cls_idx] += 1
    # print('true:', true_region, ', pred:', regions[cls_idx])
    y_test.append(regions.index(true_region))
    y_pred_list.append(cls_idx)
    count += 1
    # print(count)

print("recognition accuracy: ", 100 * true_labels/len(df))
print(classification_report(y_test, y_pred_list, target_names=regions))

for classname, correct_count in correct_pred.items():
    try:
        accuracy = 100 * float(correct_count) / total_pred[classname]
    except:
        accuracy = 0.0
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))


plt.figure(figsize=(100, 50))
df_cm = pd.DataFrame(confusion_matrix, index=regions, columns=regions).astype(int)
heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig("conf_matrix.png")
