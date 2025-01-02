import pandas as pd
import os
from glob import glob
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

f_path = '/data_tb/baseData/all_new.csv'
df = pd.read_csv(f_path)

train, test = train_test_split(df, test_size=0.1)
stop = 1

# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
# X = df[["image_path", "car_labels"]].values
# y = df["region"].values
# sss.get_n_splits(X, y)
# for train_index, test_index in sss.split(X, y):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# train = np.concatenate([X_train, y_train], axis=1)
# test = np.concatenate([X_test, y_test], axis=1)
#
df_train = pd.DataFrame(data=train.values, columns=['filename', 'label'])
df_val = pd.DataFrame(data=test.values, columns=['filename', 'label'])
stop = 1
#
# print(df_train["region"].value_counts())
# print(df_val["region"].value_counts())
#
df_train.to_csv(f_path.replace('all_new.csv', 'train_new.csv'), index_label=None, index=None)
df_val.to_csv(f_path.replace('all_new.csv', 'test_new.csv'), index_label=None, index=None)





# prefix = '/mnt/sdb1/LP_RECOGNIZER_DATA/'
# base_folder = '/data_tb/fullFacilityData/carCiy'

# poland_images = glob(os.path.join(base_folder, '**', "*"))
# turkey_images = glob(os.path.join(base_folder,  '**', "**", "*"))

# data = []

# for idx, item in enumerate(poland_images):
#     print()
#     plate_image = glob(os.path.join(item, "plate.jpg"))
#     label_file = glob(os.path.join(item, "plate.txt"))
#
#     if len(plate_image) == 1 and len(label_file) == 1:
#         plate_image = plate_image[0].replace(prefix, '')
#         label_file = label_file[0]
#
#         with open(label_file, 'r') as f:
#             label = f.read()
#         sample = [plate_image, label]
#         data.append(sample)
# df = pd.DataFrame(data=np.array(data), columns=['filename','label'])
# df.to_csv('/mnt/sdb1/LP_RECOGNIZER_DATA/data/poland/poland.csv', index_label=False, index=False)

# for idx, folder in enumerate(turkey_images):
#     print(idx)
#     synthetic_image = glob(os.path.join(folder, "synthetic.jpg"))
#     real_image = glob(os.path.join(folder, 'real.jpg'))
#     label_file = glob(os.path.join(folder, 'plate.txt'))
#
#     if len(synthetic_image) == 1 and len(label_file) == 1:
#         label_file = label_file[0]
#         synthetic_image = synthetic_image[0].replace(prefix, '')
#         with open(label_file, 'r') as f:
#             label = f.read()
#         sample = [synthetic_image, label]
#         data.append(sample)
#         if len(real_image) == 1:
#             real_image = real_image[0].replace(prefix, '')
#             sample = [real_image, label]
#             data.append(sample)
# df = pd.DataFrame(data=np.array(data), columns=['image_path','car_labels'])
# df.to_csv('/mnt/sdb1/LP_RECOGNIZER_DATA/data/turkey/turkey_plates.csv', index_label=False, index=False)

