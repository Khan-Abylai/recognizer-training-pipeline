import copy
import shutil
import time
import os
import albumentations as A
import cv2
import numpy as np
import torch
import glob
from config import base_config as config
from utils.converter import StrLabelConverter
from dataset.utils import preprocess
import onnxruntime as onnxrt

DEVICE_NAME = 'cpu'
onnx_session = onnxrt.InferenceSession("../weights/recognition_model_uae_iteration_4.onnx",
                                       providers=['CPUExecutionProvider'])
converter = StrLabelConverter(config.alphabet)
transformer = A.Compose([A.NoOp()])
regions = ["dubai", "abu-dhabi", "sharjah", "ajman", "ras-al-khaimah", "fujairah", "alquwain"]
out_folder = os.path.join(f'../debug/exp3')

images = sorted(glob.glob('../debug/image2/*.jpg'))
for idx, image in enumerate(images):
    image_path = copy.deepcopy(image)
    start_time = time.time()
    img = cv2.imread(image)
    preprocessed_image = preprocess(img, transform=transformer).unsqueeze(0).contiguous()
    binding = onnx_session.io_binding()
    binding.bind_input(name='actual_input', device_type=DEVICE_NAME, device_id=0, element_type=np.float32,
                       shape=tuple(preprocessed_image.shape), buffer_ptr=preprocessed_image.data_ptr(), )
    z_tensor = torch.empty((1, 38, 37), dtype=torch.float32, device='cpu:0').contiguous()
    binding.bind_output(name='output', device_type=DEVICE_NAME, device_id=0, element_type=np.float32,
                        shape=tuple(z_tensor.shape), buffer_ptr=z_tensor.data_ptr(), )
    x_tensor = torch.empty((1, 7), dtype=torch.float32, device='cpu:0').contiguous()

    binding.bind_output(name='output2', device_type=DEVICE_NAME, device_id=0, element_type=np.float32,
                        shape=tuple(x_tensor.shape), buffer_ptr=x_tensor.data_ptr(), )
    onnx_session.run_with_iobinding(binding)
    predictions = z_tensor
    cls = x_tensor
    stop = 1
    cls_idx = cls.argmax(1).item()
    predictions = predictions.permute(1, 0, 2).contiguous()
    prediction_size = torch.IntTensor([predictions.size(0)]).repeat(1)
    predicted_probs, predicted_labels = predictions.detach().cpu().max(2)
    predicted_probs = np.around(torch.exp(predicted_probs).permute(1, 0).numpy(), decimals=1)
    predicted_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=False))
    predicted_raw_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=True))
    new_path = os.path.join(out_folder, predicted_test_labels.item() + '_' + regions[cls_idx] + '.jpg')
    shutil.copy(image_path, new_path)
