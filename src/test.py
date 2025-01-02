import albumentations as A
import cv2
import numpy as np
import torch
import glob
from models.base_model import CRNN_2
from config import base_config as config
from utils.converter import StrLabelConverter
from dataset.utils import preprocess

if __name__ == '__main__':
    model = CRNN_2(image_h=config.img_h, num_class=config.num_class, num_layers=config.model_lstm_layers,
                   is_lstm_bidirectional=config.model_lsrm_is_bidirectional, num_regions=config.num_regions)
    converter = StrLabelConverter(config.alphabet)
    state = torch.load('/home/user/parking_recognizer/weights/epoch_499.pth')
    state_dict = state['state_dict']

    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()
    regions = ["dubai", "abu-dhabi", "sharjah", "ajman", "ras-al-khaimah", "fujairah", "alquwain"]
    transformer = A.Compose([A.NoOp()])

    images = sorted(glob.glob('/home/user/parking_recognizer/debug/test_images_uae/*'))
    for idx, image in enumerate(images):
        img = cv2.imread(image)
        preprocessed_image = preprocess(img, transform=transformer).unsqueeze(0)

        cuda_image = preprocessed_image.cuda()

        predictions, cls = model(cuda_image)
        cls_idx = cls.argmax(1).item()
        predictions = predictions.permute(1, 0, 2).contiguous()
        prediction_size = torch.IntTensor([predictions.size(0)]).repeat(1)
        predicted_probs, predicted_labels = predictions.detach().cpu().max(2)
        predicted_probs = np.around(torch.exp(predicted_probs).permute(1, 0).numpy(), decimals=1)
        predicted_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=False))
        predicted_raw_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=True))

        print(image, predicted_test_labels, regions[cls_idx])
