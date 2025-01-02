import albumentations as A
import cv2
import numpy as np
import torch
import glob
from models.base_model import CRNN_3
from config import base_config as config
from utils.converter import StrLabelConverter
from dataset.utils import preprocess
import time
import os

if __name__ == '__main__':
    model = CRNN_3(img_channel=3, img_height=32, img_width=128, num_class=config.num_class, map_to_seq_hidden=64,
                   rnn_hidden=256, leaky_relu=False)
    device = torch.device("cuda:1")
    model = torch.nn.parallel.DataParallel(model, device_ids=[1])
    converter = StrLabelConverter(config.alphabet)
    state = torch.load("../weights/azer/crnn_lite_azer.pth")
    state_dict = state['state_dict']

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transformer = A.Compose([A.NoOp()])

    dummy_input = torch.rand((1,3,32,128), requires_grad=False)

    _ = model(dummy_input)

    images = sorted(glob.glob('/home/yeleussinova/data_1TB/azer/plates/*'))
    print(f'images: {len(images)}')
    count = 0
    for idx, image in enumerate(images):
        img = cv2.imread(image)

        preprocessed_image = preprocess(img, transform=transformer).unsqueeze(0)

        cuda_image = preprocessed_image.cuda()

        start_time = time.time()

        predictions = model(cuda_image)
        predictions = predictions.log_softmax(2)
        end_time = time.time()

        exec_time = end_time - start_time

        prediction_size = torch.IntTensor([predictions.size(0)]).repeat(1)
        predicted_probs, predicted_labels = predictions.detach().cpu().max(2)
        predicted_probs = np.around(torch.exp(predicted_probs).permute(1, 0).numpy(), decimals=1)
        predicted_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=False))
        predicted_raw_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=True))

        print(predicted_test_labels, np.mean(predicted_probs), image)
        # label = os.path.basename(image).split("_")[0].lower()
        # if label != str(predicted_test_labels):
        #     pass
