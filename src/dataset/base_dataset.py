from abc import ABCMeta, abstractmethod

import pandas as pd
try:
    from src.dataset.utils import preprocess
except:
    from dataset.utils import preprocess
from torch.utils.data.dataset import Dataset


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, csv_files, image_w, image_h, transform=None, train=True, return_filepath=True):
        self.transform = transform

        if len(csv_files) == 1:
            chunks = pd.read_csv(csv_files[0], chunksize=100000)
            df = pd.concat(chunks)
        else:
            df = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files])

        if train:
            df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        self.df = df
        if self.df.shape[1] == 1:
            raise NotImplementedError
        if self.df.shape[1] == 2:
            self.df.columns = ['filename', 'label']
        if self.df.shape[1] == 3:
            self.df.columns = ['filename', 'label', 'region']
        self.img_paths = df['filename'].values
        self.labels = df['label'].values
        self.train = train
        self.return_filepath = return_filepath
        self.image_w = image_w
        self.image_h = image_h

    def __len__(self):
        return len(self.img_paths)

    @abstractmethod
    def get_image(self, index):
        pass

    def preprocess_x(self, x):
        if self.train:
            x = preprocess(x, self.image_w, self.image_h, transform=self.transform)
        else:
            x = preprocess(x, self.image_w, self.image_h)
        return x

    @abstractmethod
    def __getitem__(self, index):
        x = self.get_image(index)
        if x is None:
            return self[index + 1]

        image = self.preprocess_x(x)
        label = self.labels[index]

        label = str(label).strip().lower() \
            .replace('\n', '') \
            .replace(' ', '') \
            .replace('!', '') \
            .replace('#', '') \
            .replace('@', '') \
            .replace('?', '') \
            .replace('$', '') \
            .replace('-', '') \
            .replace('.', '') \
            .replace('|', '') \
            .replace('_', '') \
            .replace('/', '') \
            .replace("*", '') \
            .replace('=', '').encode("ascii", "ignore").decode()
        whitelist = set('abcdefghijklmnopqrstuvwxyz1234567890')
        label = ''.join(filter(whitelist.__contains__, label))
        if self.return_filepath:
            return image, label, self.img_paths[index]
        else:
            return image, label