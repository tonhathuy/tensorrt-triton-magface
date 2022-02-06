import numpy as np
import onnx
import torch

from inference.network_inf import builder_inf

def convert_onnx(net, output, opset=11, simplify=False):
    assert isinstance(net, torch.nn.Module)
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
    img = img.astype(np.float)
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    dummy_input = torch.zeros(1, 3, 112, 112, device='cuda')
    
    net.eval()
    torch.onnx.export(net.module, dummy_input, output, input_names=['input'], output_names=['output'], 
    keep_initializers_as_inputs=False, verbose=True, opset_version=opset)
    model = onnx.load(output)
    graph = model.graph
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = '-1'

    # Checks
    onnx_model = onnx.load(output)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model

    if simplify:
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, output)
    print('ONNX export success, saved as %s' % output)

    y = net(dummy_input)
    return y

    
if __name__ == '__main__':
    import os
    import argparse
    

    parser = argparse.ArgumentParser(description='Magface PyTorch to onnx')
    parser.add_argument('--output', type=str, default='./inference/models/magface_iresnet50_MS1MV2_dp.onnx', help='output onnx path')
    parser.add_argument('--simplify', type=bool, default=False, help='onnx simplify')
    parser.add_argument('--onnx_infer', action='store_true', default=True, help='onnx infer test')
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
    parser.add_argument('--resume', default='./inference/models/magface_iresnet50_MS1MV2_dp.pth', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--cpu-mode', action='store_true', help='Use the CPU.')
    parser.add_argument('--dist', default=1, help='use this if model is trained with dist')
    args = parser.parse_args()
    # input_file = args.resume
    # if os.path.isdir(input_file):
    #     input_file = os.path.join(input_file, "model.pt")
    # assert os.path.exists(input_file)
    # model_name = os.path.basename(os.path.dirname(input_file)).lower()
    # params = model_name.split("_")
    # if len(params) >= 3 and params[1] in ('arcface', 'cosface'):
    #     if args.network is None:
    #         args.network = params[2]
    assert args.arch is not None
    print(args)
    print('=> modeling the network ...', 'green')
   
    model = builder_inf(args)
    backbone_onnx = torch.nn.DataParallel(model)
    if not args.cpu_mode:
        model = model.cuda()
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "model.onnx")
    y = convert_onnx(backbone_onnx, args.output, simplify=args.simplify)

    # onnx infer
    if args.onnx_infer:
        import onnxruntime 
        providers =  ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(args.output, providers=providers)
        
        dummy_input = torch.zeros(1, 3, 112, 112, device='cuda')
        im = dummy_input.cpu().numpy().astype(np.float32) # torch to numpy
        y_onnx = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: im})[0]
        print("pred's onnx shape is ",y_onnx.shape)
        print("pred's pt shape is ",y.data.cpu().numpy().shape)
        print("max(|torch_pred - onnx_pred|） =",abs(y.data.cpu().numpy()-y_onnx).max())
    
