import time
import os
import albumentations as A
import cv2
import numpy as np
import torch
import glob
from models.base_model import CRNN
from config import base_config as config
from utils.converter import StrLabelConverter
from dataset.utils import preprocess
import torch
import onnxruntime as onnxrt


DEVICE_NAME = 'cpu'
converter = StrLabelConverter(config.alphabet)
transformer = A.Compose([A.NoOp()])

# onnx_session= onnxrt.InferenceSession("../weights/recognizer_base1.onnx", providers=['CPUExecutionProvider'])#
# for idx, image in enumerate(images):
#     start_time = time.time()
#     img = cv2.imread(image)
#     x = cv2.resize(img, (128, 32))
#     x = x.astype(np.float32) / 255.
#     x = x.transpose(2, 0, 1)
#     x = torch.tensor(x)
#
#     # preprocessed_image = preprocess(img, transform=transformer).unsqueeze(0).contiguous()
#     preprocessed_image = x.unsqueeze(0).contiguous()
#     binding = onnx_session.io_binding()
#     binding.bind_input(
#         name='actual_input',
#         device_type=DEVICE_NAME,
#         device_id=0,
#         element_type=np.float32,
#         shape=tuple(preprocessed_image.shape),
#         buffer_ptr=preprocessed_image.data_ptr(),
#     )
#     z_tensor = torch.empty((1, 30, 37), dtype=torch.float32, device='cpu:0').contiguous()
#     binding.bind_output(
#         name='output',
#         device_type=DEVICE_NAME,
#         device_id=0,
#         element_type=np.float32,
#         shape=tuple(z_tensor.shape),
#         buffer_ptr=z_tensor.data_ptr(),
#     )
#     onnx_session.run_with_iobinding(binding)
#     predictions = z_tensor
#     predictions = predictions.permute(1, 0, 2).contiguous()
#     # for idx,item in enumerate(predictions):
#     #     for value in item[0]:
#     #         print(str(value.item())[:5], end=",")
#     #     print()
#     prediction_size = torch.IntTensor([predictions.size(0)]).repeat(1)
#     predicted_probs, predicted_labels = predictions.detach().cpu().max(2)
#     predicted_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=False))
#     predicted_raw_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=True))
#
#     end_time = time.time() - start_time
#     print(f'base exec time: {end_time}')
#     print(predicted_test_labels)


#resnet test
# onnx_session2= onnxrt.InferenceSession("../weights/recognizer_reznet2.onnx", providers=['CPUExecutionProvider'])
# for idx, image in enumerate(images):
#     start_time = time.time()
#     img = cv2.imread(image)
#     preprocessed_image = preprocess(img, transform=transformer).unsqueeze(0).contiguous()
#     binding2 = onnx_session2.io_binding()
#     binding2.bind_input(
#         name='input',
#         device_type=DEVICE_NAME,
#         device_id=0,
#         element_type=np.float32,
#         shape=tuple(preprocessed_image.shape),
#         buffer_ptr=preprocessed_image.data_ptr(),
#     )
#     z_tensor = torch.empty((10, 1, 37), dtype=torch.float32, device='cpu:0').contiguous()
#     binding2.bind_output(
#         name='output',
#         device_type=DEVICE_NAME,
#         device_id=0,
#         element_type=np.float32,
#         shape=tuple(z_tensor.shape),
#         buffer_ptr=z_tensor.data_ptr(),
#     )
#     onnx_session2.run_with_iobinding(binding2)
#     predictions = z_tensor
#     predictions = predictions.log_softmax(2)
#     prediction_size = torch.LongTensor([predictions.size(0)]).repeat(1)
#     predictions = predictions.argmax(2).detach().cpu()
#     predicted_train_labels = np.array(converter.decode(predictions, prediction_size, raw=False))
#     end_time = time.time() - start_time
#     print(f'resnet exec time: {end_time} labels: {predicted_train_labels}')

# test crnn3
images = sorted(glob.glob('../img/plates/*'))

onnx_session3 = onnxrt.InferenceSession("../weights/crnn-lite-sim.onnx", providers=['CPUExecutionProvider'])
for idx, image in enumerate(images):
    start_time = time.time()
    img = cv2.imread(image)
    preprocessed_image = preprocess(img, transform=transformer).unsqueeze(0).contiguous()
    stop = 1
    binding = onnx_session3.io_binding()
    # print(preprocessed_image.shape)
    # print(preprocessed_image[0][0][0])
    binding.bind_input(
        name='actual_input',
        device_type=DEVICE_NAME,
        device_id=0,
        element_type=np.float32,
        shape=tuple(preprocessed_image.shape),
        buffer_ptr=preprocessed_image.data_ptr(),
    )
    z_tensor = torch.empty((31, 1, 38), dtype=torch.float32, device='cpu:0').contiguous()
    binding.bind_output(
        name='output',
        device_type=DEVICE_NAME,
        device_id=0,
        element_type=np.float32,
        shape=tuple(z_tensor.shape),
        buffer_ptr=z_tensor.data_ptr(),
    )
    onnx_session3.run_with_iobinding(binding)
    predictions = z_tensor
    predictions = predictions.log_softmax(2)
    # print(predictions.shape)
    prediction_size = torch.IntTensor([predictions.size(0)]).repeat(1)
    predicted_probs, predicted_labels = predictions.detach().cpu().max(2)
    # predicted_labels = predictions.detach().cpu().max(2)[1]
    predicted_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=False))
    predicted_raw_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=True))
    predicted_probs = np.around(torch.exp(predicted_probs).permute(1, 0).numpy(), decimals=1)
    print(predicted_test_labels,  np.mean(predicted_probs))