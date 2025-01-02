import albumentations as A

from augmentation.transforms import transform_old

train_csv = ['/europe_last/train.csv']

val_csv = ['/europe_last/val.csv']

params = {'train': {'transform': transform_old, 'train': True, 'csv_files': train_csv},
          'val': {'transform': A.Compose([A.NoOp()]), 'train': False, 'csv_files': val_csv}}
