import numpy as np
import torch
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.nn as nn
import string
from models import CRNN

parser = argparse.ArgumentParser()
parser.add_argument('--weights_path', type=str,
                    default='/workspace/rec/weights_final_final/64_100_Train:_8.3789,_Accuracy:_0.9903,_Val:_0.8664,_Accuracy:_0.9992,_lr:_1e-05.pth')
parser.add_argument('--out_path', type=str, default='/workspace/rec/azer_recognizer_final.np')
parser.add_argument('--cfg_file', type=str, default='train_config.yml')
args = parser.parse_args()

if __name__ == '__main__':
    import yaml
    with open(args.cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
    
    model = CRNN(**cfg['model']).cuda(0)
    new_st = {}
    state_dict = torch.load(args.weights_path)['state_dict']
    for k, v in state_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        new_st[k] = v

    model.load_state_dict(new_st)
    model.eval()
    model.cpu()

    print(model)


    s_dict = model.state_dict()
    total = 0
    t = 'num_batches_tracked'
    np_weights = np.array([], dtype=np.float32)
    for k, v in s_dict.items():
        if k[-len(t):] == t:
            continue
        total += v.numel()
        v_reshape = v.reshape(-1)
        np_v = v_reshape.data.numpy()
        np_weights = np.concatenate((np_weights, np_v))

    print(total)
    print(np_weights.shape)
    print(np_weights)
    print(np_weights.dtype)

    np_weights.tofile(args.out_path)