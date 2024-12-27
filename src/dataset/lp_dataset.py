import os

import cv2
try:
    from src.dataset.base_dataset import BaseDataset
    from src.augmentation.transforms import transform_old
except:
    from .base_dataset import BaseDataset
    from ..augmentation.transforms import transform_old

class LPDataset(BaseDataset):
    def __init__(self, csv_files, data_dir='', use_region=False, **kwargs):
        self.data_dir = data_dir
        csv_files = [os.path.join(data_dir, csv_file) for csv_file in csv_files]
        super(LPDataset, self).__init__(csv_files=csv_files, **kwargs)
        self.use_region = use_region
        if use_region:
            self.regions = self.df['region']

    def get_image(self, index):
        path = self.img_paths[index % self.__len__()]
        img = cv2.imread(os.path.join(self.data_dir, path))
        return img

    def __getitem__(self, index):
        x = self.get_image(index)
        if x is None:
            return self[index + 1]

        image = self.preprocess_x(x)
        label = self.labels[index]
        region = ''
        if self.use_region:
            region = self.regions[index]
        whitelist = set('abcdefghijklmnopqrstuvwxyz1234567890')
        label = str(label).strip().lower().replace('\n', '').replace(' ', '').replace('!', '').replace('#', '').replace('@', '').replace('?', '').replace('$', '').replace('-', '').replace('.', '').replace('|', '').replace('_','').replace('`', '').replace('=', '').encode("ascii", "ignore").decode()
        label = ''.join(filter(whitelist.__contains__, label))
        if self.return_filepath:
            return image, label, region, self.img_paths[index]
        else:
            return image, label, region


if __name__ == '__main__':
    params = {"train": True, "data_dir": "/mnt/data",
              "csv_files": ["/mnt/data/csv/mini_train.csv"], "transform": transform_old
              }
    lpRegionDataset = LPDataset(**params)
    debug_dir = "/home/user/parking_recognizer/debug/test_images"
    for idx, sample in enumerate(lpRegionDataset):
        print(idx)
