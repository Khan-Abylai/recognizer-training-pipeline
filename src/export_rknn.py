import sys
from rknn.api import RKNN
import argparse

DATASET_PATH = '../model/plate_dataset.txt'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="../weights/sng_crnn_lite.onnx")
    parser.add_argument('--platform', type=str, default="rk3588")
    parser.add_argument('--do_quant', type=bool, default=False)
    parser.add_argument('--output_path', type=str, default="../weights/sng_crnn_lite_with_prep.rknn")
    parser.add_argument('--dataset_path', type=str, default="dataset.txt")
    args = parser.parse_args()

    # model_path, platform, do_quant, output_path = parse_arg()

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0.0, 0.0, 0.0]], std_values=[[1/255.0, 1/255.0, 1/255.0]], target_platform=args.platform)
    # rknn.config(target_platform=args.platform)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=args.model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=args.do_quant, dataset=args.dataset_path)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(args.output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()