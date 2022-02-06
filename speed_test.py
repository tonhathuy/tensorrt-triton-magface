from inference.network_inf import builder_inf
from export_trt import TrtModel
import argparse
import torch
import time
from tqdm import tqdm


def run(model,img,warmup_iter,iter):
    
    
    print('start warm up...')
    for _ in tqdm(range(warmup_iter)):
        model(img) 
    
   
    print('start calculate...')
    torch.cuda.synchronize()
    start = time.time()
    for __ in tqdm(range(iter)):
        model(img) 
        torch.cuda.synchronize()
    end = time.time()
    return ((end - start) * 1000)/float(iter)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_path', type=str,required=True, help='torch weights path')  
    parser.add_argument('--trt_path', type=str,required=True, help='tensorrt weights path')

    parser.add_argument('--device', type=int,default=0, help='cuda device')
    parser.add_argument('--img_shape', type=list,default=[1,3,112,112], help='tensorrt weights path')
    parser.add_argument('--warmup_iter', type=int, default=100,help='warm up iter')  
    parser.add_argument('--iter', type=int, default=500,help='average elapsed time of iterations')  
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
    parser.add_argument('--resume', default='./inference/models/mag-cosface_iresnet50_MS1MV2_ddp_fp32.pth', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--cpu-mode', action='store_true', help='Use the CPU.')
    parser.add_argument('--dist', default=1, help='use this if model is trained with dist')
    opt = parser.parse_args()

    img = torch.zeros(opt.img_shape)
    opt.resume = opt.torch_path
    # -----------------------torch-----------------------------------------
    model = builder_inf(opt)
    model = torch.nn.DataParallel(model)
    model.eval()
    total_time=run(model.to(opt.device),img.to(opt.device),opt.warmup_iter,opt.iter)
    print('Pytorch is  %.2f ms/img'%total_time)

    # -----------------------tensorrt-----------------------------------------
    model=TrtModel(opt.trt_path)
    total_time=run(model,img.numpy(),opt.warmup_iter,opt.iter)
    model.destroy()
    print('TensorRT is  %.2f ms/img'%total_time)