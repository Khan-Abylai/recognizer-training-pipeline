import pandas as pd
import os
from glob import glob
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

f_path = '/home/yeleussinova/data_SSD/eu_plate_data/train.csv'
df = pd.read_csv(f_path)
print(df.tail(1))
# df["region"] = ['albania', 'andorra', 'austria', 'belgium', 'bosnia', 'bulgaria', 'croatia', 'cyprus', 'czech', 'estonia',
#            'finland', 'france', 'germany', 'greece', 'hungary', 'ireland', 'italy', 'latvia',
#            'licht', 'lithuania', 'luxemburg', 'malta', 'monaco', 'montenegro', 'netherlands', 'poland',
#            'portugal', 'romania', 'san_marino', 'slovakia', 'slovenia', 'spain', 'sweden', 'swiss']
stop = 1
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
X = df[["image_path", "plate_label"]].values
y = df["region"].values

sss.get_n_splits(X, y)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
train = np.concatenate([X_train, y_train], axis=1)
test = np.concatenate([X_test, y_test], axis=1)

df_train = pd.DataFrame(data=train, columns=['image_path', 'car_labels', 'region'])
df_val = pd.DataFrame(data=test, columns=['image_path', 'car_labels', 'region'])

print(df_train["region"].value_counts())
print(df_val["region"].value_counts())

df_train.to_csv("/home/yeleussinova/data_SSD/eu_plate_data/train1.csv", index_label=None, index=None)
df_val.to_csv("/home/yeleussinova/data_SSD/eu_plate_data/test.csv", index_label=None, index=None)