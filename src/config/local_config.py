import albumentations as A

from augmentation.transforms import transform_old

train_csv = ['/workspace/data/uae/uae_iteration_5/after_cleaning_train.csv']

val_csv = ['/workspace/data/uae/uae_iteration_5/after_cleaning_test.csv']

params = {'train': {'transform': transform_old, 'train': True, 'csv_files': train_csv},
          'val': {'transform': A.Compose([A.NoOp()]), 'train': False, 'csv_files': val_csv}}
