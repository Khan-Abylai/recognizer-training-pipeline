import numpy as np
import torch

if __name__ == '__main__':
    in_path = '/home/user/parking_recognizer/weights/epoch_499.pth'
    # in_path = './weights/ckpt.pth'
    out_path = '/home/user/parking_recognizer/weights/recognizer_weights_usa_last.np'
    torch_weights = torch.load(in_path, map_location='cpu')['state_dict']
    stop = 1
    weights = []

    for k, v in torch_weights.items():
        if 'num_batches_tracked' in k:
            continue
        weights += v.flatten().cpu().tolist()
        print(k, "current weights size:", len(v.flatten().cpu().tolist()), 'full size: ', len(weights))
        # print('\n')
    print(len(weights))
    weights = np.array(weights, dtype=np.float32)
    weights.tofile(out_path)
