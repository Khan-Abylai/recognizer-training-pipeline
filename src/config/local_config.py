import albumentations as A

from augmentation.transforms import transform_old



train_csv = ['/home/user/data/experiment/fine_tuning_for_UAE/uae_data/train.csv']

val_csv = ['/home/user/data/experiment/fine_tuning_for_UAE/uae_data/val.csv']

params = {
    'train': {
        'transform': transform_old,
        'train': True,
        'csv_files': train_csv
    },
    'val': {
        'transform': A.Compose([A.NoOp()]),
        'train': False,
        'csv_files': val_csv
    }
}
