import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import argparse
import time

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()

# def build_engine(onnx_file_path):
#     # initialize TensorRT engine and parse ONNX model
#     builder = trt.Builder(TRT_LOGGER)
#     network = builder.create_network()
#     parser = trt.OnnxParser(network, TRT_LOGGER)
    
#     # parse ONNX
#     with open(onnx_file_path, 'rb') as model:
#         print('Beginning ONNX file parsing')
#         parser.parse(model.read())
#     print('Completed parsing of ONNX file')

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
def GiB(val):
    return val * 1 << 30

def ONNX_to_TRT(onnx_model_path=None,trt_engine_path=None,fp16_mode=False):
    """
    TensorRT V8 
    
    fp16_mode: True fp16
    onnx_model_path: onnx
    trt_engine_path: trt
    """
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    config = builder.create_builder_config()
    config.max_workspace_size=GiB(1) 
    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16) 
    with open(onnx_model_path, 'rb') as model:
        assert parser.parse(model.read())
        serialized_engine=builder.build_serialized_network(network, config)

   
    with open(trt_engine_path, 'wb') as f:
        f.write(serialized_engine)  # 序列化

    print('TensorRT file in ' + trt_engine_path)
    print('============ONNX->TensorRT SUCCESS============')

class TrtModel():
    '''
    TensorRT infer
    '''
    def __init__(self,trt_path):
        self.ctx=cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        print(trt_path)
        with open(trt_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size
    
    def __call__(self,img_np_nchw):
        '''
        TensorRT
        :param img_np_nchw
        '''
        self.ctx.push()

        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        np.copyto(host_inputs[0], img_np_nchw.ravel())
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        self.ctx.pop()
        return host_outputs[0]


    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./inference/models/magface_iresnet18_casia_dp.onnx', help='weights path')  # from yolov5/models/
    parser.add_argument('--img_size', nargs='+', type=int, default=[112, 112], help='image size')  # height, width
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--simplify', action='store_true', default=False, help='simplify onnx')
    parser.add_argument('--dynamic', action='store_true', default=False, help='enable dynamic axis in onnx model')
    parser.add_argument('--onnx2pb', action='store_true', default=False, help='export onnx to pb')
    parser.add_argument('--onnx_infer', action='store_true', default=True, help='onnx infer test')
    #=======================TensorRT=================================
    parser.add_argument('--onnx2trt', action='store_true', default=False, help='export onnx to tensorrt')
    parser.add_argument('--fp16_trt', action='store_true', default=True, help='fp16 infer')
    #================================================================
    opt = parser.parse_args()
    f = opt.weights
    print('\nStarting TensorRT...')
    # ONNX_to_TRT(onnx_model_path=f,trt_engine_path=f.replace('.onnx', '.trt'),fp16_mode=opt.fp16_trt)
    model=TrtModel('./model_v3.trt')
    img_np_nchw = np.random.rand(1, 3, *opt.img_size).astype(np.float32)
    time_start=time.time()
    pred=model(img_np_nchw) # forward
    print('time: ',time.time()-time_start)
    print('pred.shape: ', pred.shape)
    # pred=model(img.numpy()).reshape(opt.output_shape) # forward
    model.destroy()