import torch
from models.base_model import CRNN_2
from config import base_config as config
from utils.converter import StrLabelConverter

checkpoint_path = '../weights/recognition_model_uae_iteration_4.pth'

model = CRNN_2(image_h=config.img_h, num_class=config.num_class, num_layers=config.model_lstm_layers,
               is_lstm_bidirectional=config.model_lsrm_is_bidirectional, num_regions=7)
model = torch.nn.parallel.DataParallel(model)
converter = StrLabelConverter(config.alphabet)
state = torch.load(checkpoint_path)
state_dict = state['state_dict']

model.load_state_dict(state_dict)
model.cuda()
model.eval()

if isinstance(model, torch.nn.DataParallel):
    model = model.module

model.eval()
model.cpu()
dummy_input = torch.randn(1, 3, 64, 160).cpu()
input_names = ["actual_input"]
output_names = ["output", "output2"]

torch.onnx.export(model, dummy_input, checkpoint_path.replace('.pth', '.onnx'), verbose=True, input_names=input_names,
                  output_names=output_names, export_params=True, opset_version=11)
print("Model converted to the onnx")
