CUDA=$1
WEIGHT_DIR=$2
BACKBONE=$3
MODEL_NAME= $4
# CUDA=0
# WEIGHT_DIR=./weights
# BACKBONE=iresnet100
# MODEL_NAME= magface_iresnet100_MS1MV2_dp
MODEL_PTH_PATH = ./weights/$MODEL_NAME.pth
MODEL_ONNX_PATH = ./weights/$MODEL_NAME.onnx
MODEL_TRT_PATH = ./weights/$MODEL_NAME.trt
# covert onnx
CUDA_VISIBLE_DEVICES=$CUDA python export_onnx.py --arch $BACKBONE --resume $MODEL_PTH_PATH --output $MODEL_ONNX_PATH --fp16_trt
# convert trt
trtexec --onnx=$MODEL_ONNX_PATH --fp16 --device=2 --best --minShapes=input:1x3x112x112 --optShapes=input:2x3x112x112 --maxShapes=input:16x3x112x112 --workspace=4096 --verbose --saveEngine=$MODEL_TRT_PATH --explicitBatch


mkdir -p ./model_repository/magface_trt/1/
cp $MODEL_TRT_PATH ./model_repository/magface_trt/1/model.plan

mkdir -p ./model_repository/magface_trt/1/
cp $MODEL_ONNX_PATH ./model_repository/magface_onnx/1/model.onnx

# # TensorRT Inference
# CUDA_VISIBLE_DEVICES=$CUDA python torch2trt/main.py --trt_path ./$WEIGHT_DIR/yolov5n-face.trt

# # Speed test
# CUDA_VISIBLE_DEVICES=$CUDA python 
