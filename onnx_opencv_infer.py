import cv2
import numpy as np
 
onnx_model_path = "./inference/models/magface_iresnet18_casia_dp.onnx"
sample_image = "./inference/toy_imgs/0.jpg"
 
net =  cv2.dnn.readNetFromONNX(onnx_model_path) 
image = cv2.imread(sample_image)

dummy_input = np.ones((112, 112, 3), dtype=np.float32)
blob = cv2.dnn.blobFromImage(dummy_input, 1.0, (112, 112),(0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
preds = net.forward()
print(preds.shape, preds[0][:10], preds.dtype)

check = True

if check:
    import torch
    import argparse
    from inference.network_inf import builder_inf
    parser = argparse.ArgumentParser(description='Magface PyTorch to onnx')
    parser.add_argument('--arch', default='iresnet18', type=str,
                        help='backbone architechture')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--embedding_size', default=512, type=int,
                        help='The embedding feature size')
    parser.add_argument('--resume', default='./inference/models/magface_iresnet18_casia_dp.pth', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--cpu-mode', action='store_true', help='Use the CPU.')
    parser.add_argument('--dist', default=1, help='use this if model is trained with dist')
    args = parser.parse_args()
    model = builder_inf(args)
    model = torch.nn.DataParallel(model)
    model.eval()
    # print(image.shape)
    dummy_input = torch.ones(1, 3, 112, 112, device='cuda')
    preds_pt = model(dummy_input)
    print(preds_pt.shape, preds_pt[0][:10], preds_pt.dtype)
    print("max(|torch_pred - onnx_pred|） =",abs(preds_pt.data.cpu().numpy()-preds).max())