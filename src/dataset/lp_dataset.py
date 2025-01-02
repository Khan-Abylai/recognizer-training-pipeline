import os

import cv2
from src.dataset.base_dataset import BaseDataset
from src.augmentation.transforms import transform_old


class LPDataset(BaseDataset):
    def __init__(self, data_dir, csv_files, **kwargs):
        self.data_dir = data_dir
        csv_files = [os.path.join(data_dir, csv_file) for csv_file in csv_files]
        super(LPDataset, self).__init__(csv_files=csv_files, **kwargs)

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

        label = str(label).strip().lower().replace('\n', '').replace(' ', '').replace('!', '').replace('#', '').replace(
            '@', '').replace('?', '').replace('$', '').replace('-', '').replace('.', '').replace('|', '').replace('_',
                                                                                                                  '').replace(
            '=', '').encode("ascii", "ignore").decode()

        if self.return_filepath:
            return image, label, self.img_paths[index]
        else:
            return image, label


class LPRegionDataset(BaseDataset):
    def __init__(self, data_dir, csv_files, **kwargs):
        self.data_dir = data_dir
        csv_files = [os.path.join(data_dir, csv_file) for csv_file in csv_files]
        super(LPRegionDataset, self).__init__(csv_files=csv_files, **kwargs)
        self.regions = self.df["region"]

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
        region = self.regions[index]
        whitelist = set('abcdefghijklmnopqrstuvwxyz1234567890@&!?%^#$|')
        label = str(label).strip().lower().replace('\n', '').replace(' ', '').replace('!', '').replace('#', '').replace(
            '@', '').replace('?', '').replace('$', '').replace('-', '').replace('.', '').replace('|', '').replace('_','').replace(
            '`', '').replace('=', '') #.encode("ascii", "ignore").decode()

        # replace acuts
        label = label.replace('å', '@').replace('ä', '&').replace('ć', '!').replace('č', '?').replace('đ', '%').replace('ö', '^').replace('ü', '#').replace('š', '$').replace('ž', '|')
        label = ''.join(filter(whitelist.__contains__, label))

        if self.return_filepath:
            return image, label, region, self.img_paths[index]
        else:
            return image, label, region


if __name__ == '__main__':
    params = {"train": True, "data_dir": "/",
              "csv_files": ["/europe_last/test.csv"], "transform": transform_old
              }

    lpRegionDataset = LPRegionDataset(**params)
    base_folder = '/home/user/recognizer_pipeline'
    debug_dir = os.path.join(base_folder, 'logs', 'exp2')
    # debug_dir = '/home/user/mnt/debug'
    for idx, sample in enumerate(lpRegionDataset):
        # print(idx)
        if idx == 1000000:
            break
