import torch
from models.base_model import CRNN, CRNN_resnet, CRNN_3
from config import base_config as config
from utils.converter import StrLabelConverter
import os
# checkpoint_path = '../weights/recognizer_base.pth'
#
# model = CRNN(image_h=config.img_h, num_class=config.num_class, num_layers=config.model_lstm_layers,
#              is_lstm_bidirectional=config.model_lsrm_is_bidirectional)
# model = torch.nn.parallel.DataParallel(model)
# converter = StrLabelConverter(config.alphabet)
# state = torch.load(checkpoint_path)
# state_dict = state['state_dict']
#
# model.load_state_dict(state_dict)
# model.cuda()
# model.eval()
#
# if isinstance(model, torch.nn.DataParallel):
#     model = model.module
#
# model.eval()
# model.cpu()
# dummy_input = torch.randn(1, 3, 32, 128).cpu()
# input_names = ["actual_input"]
# output_names = ["output"]
#
# torch.onnx.export(model, dummy_input, checkpoint_path.replace('.pth', '.onnx'), verbose=True, input_names=input_names,
#                   output_names=output_names, export_params=True, opset_version=11)
# print("Model converted to the onnx")


# resnet convert
# checkpoint_path = "/home/yeleussinova/data_SSD/uz_weights/model_uz_reznet2/69_100_Train:_0.0480,_Accuracy:_0.9101,_Val:_0.0564,_Accuracy:_0.8333,_lr:_0.0001.pth"
# resnet = CRNN_resnet(image_h=64, image_w=160, num_class=config.num_class, hl=64, is_lstm_bidirectional=True, linear_size=512)
# model = torch.nn.parallel.DataParallel(resnet)
# state = torch.load(checkpoint_path)
# state_dict = state['state_dict']
#
# model.load_state_dict(state_dict)
# model.cuda()
# model.eval()
#
# if isinstance(model, torch.nn.DataParallel):
#     model = model.module
#
# model.eval()
# model.cpu()
# h, w, c, b = 64, 160, 3, 1
# xs = torch.rand((b, c, h, w)).cpu()
# torch_out = model(xs)
# dummy_input = xs.cpu()
# input_names = ["input"]
# output_names = ["output"]
#
# torch.onnx.export(model, dummy_input, checkpoint_path.replace(".pth", ".onnx"), verbose=True, input_names=input_names,
#                   output_names=output_names, export_params=True, opset_version=11)
# print("Model converted to the onnx")
#

checkpoint_path = "../weights/crnn_lite_01.08.2024.pth"
resnet = CRNN_3(img_channel=3, img_height=32, img_width=128, num_class=config.num_class,  map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False)
model = torch.nn.parallel.DataParallel(resnet)
state = torch.load(checkpoint_path)
state_dict = state['state_dict']

model.load_state_dict(state_dict)
model.cuda()
model.eval()

if isinstance(model, torch.nn.DataParallel):
    model = model.module

model.eval()
model.cpu()
dummy_input = torch.randn(1, 3, 32, 128).cpu()
input_names = ["actual_input"]
output_names = ["output"]

torch.onnx.export(model, dummy_input, '../weights/sng_crnn_lite.onnx', verbose=True, input_names=input_names,
                  output_names=output_names, export_params=True, opset_version=11)
print("Model converted to the onnx")
