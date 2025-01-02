import os
import cv2
import json
from glob import glob
import pandas as pd
import numpy as np

annotation_path = '/home/yeleussinova/data_SSD/eu_plate_data/annotation'
data_folder = '/home/yeleussinova/data_SSD/eu_plate_data/images'
output_csv_folder = '/home/yeleussinova/data_SSD/eu_plate_data'
regions = ['albania', 'andorra', 'austria', 'belgium', 'bosnia', 'bulgaria', 'croatia', 'cyprus', 'czech', 'estonia',
           'finland', 'france', 'germany', 'greece', 'hungary', 'ireland', 'italy', 'latvia',
           'licht', 'lithuania', 'luxemburg', 'makedonia', 'malta', 'monaco', 'montenegro', 'netherlands', 'poland',
           'portugal', 'romania', 'serbia', 'san_marino', 'slovakia', 'slovenia', 'spain', 'sweden', 'swiss']

all_image_paths = np.array([x for x in glob(os.path.join(data_folder, "**", "*")) if ".jpg" in x])
df = pd.DataFrame(data=all_image_paths, columns=['image_path'])
df["basename"] = df["image_path"].apply(lambda x: os.path.basename(x))
print(df["basename"])
final = []
for image in all_image_paths:
    country = image.split('/')[-2]
    folder_id = image.split('/')[-1].replace('_synthetic.jpg', '').replace('.jpg', '')
    img_path = image.split('/')[-1]
    txt_id = image.split('/')[-1].replace('.jpg', '.txt')
    txt_path = os.path.join(annotation_path, country, 'plates', folder_id, txt_id)
    if os.path.exists(txt_path):
        f = open(txt_path, 'r')
        line = f.readline().rstrip('\n')
        label = line.split(",")[0]
        region = line.split(",")[1]

        if region in regions:
            plate_label = label.lower()
            relative_path = os.path.join('europe_last', 'images', region, img_path)
            full_data = [relative_path, plate_label, region]
            final.append(full_data)
        else:
            print(txt_path, label, region)
    else:
        print(txt_path)

df_out = pd.DataFrame(data=final, columns=['image_path', 'plate_label', 'region'])
df_out.to_csv(os.path.join(output_csv_folder, 'europe_plates.csv'), index_label=False, index=False)
