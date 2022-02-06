# tensorrt-triton-magface
Magface Triton Inferece Server Using Tensorrt

# Speed test result
'''
python speed_test.py 

|   Loss-Backbone   | Pytorch(ms) | TensorRT_FP16(ms) |
|   :------------:  | :---------: | :---------------: |
|    magface-r18    |     2.89    |        0.58       |
|    magface-r50    |     3.25    |        1.36       |
|    magface-r100   |     3.34    |        2.37       |
|    arcface-r18    |     2.90    |        0.64       |
|  mag-cosface-r50  |     6.56    |        1.34       |

### Convert Onnx -> TensorRT engine 
## Build dockerfile 
```bash 
cd tensorrt-triton-magface 
docker build -t huytn/tensorrt-20.12-py3:v1 .
```
## Run docker container 

example save weight path: ./tensorrt-triton-magface/weights/magface_iresnet100_MS1MV2_dp.pth

```bash 
docker run -it --gpus all --name tensorrt_8_magface_convert -v $(pwd):/convert/ -w /convert/ nvcr.io/nvidia/tensorrt:20.11-py3 bash
chmod +x ./convert.sh
./convert.sh 0 ./weights iresnet100 magface_iresnet100_MS1MV2_dp
```

### Speed test

```bash 
python3 speed_test.py --torch_path ./weights/magface_iresnet100_MS1MV2_dp.pth --trt_path ./weights/magface_iresnet100_MS1MV2_dp.pth
```

### Triton server 

Check if Server running correctly:
```bash 
$ curl -v localhost:8330/v2/health/ready
...
< HTTP/1.1 200 OK
< Content-Length: 0
< Content-Type: text/plain
```

Run docker image of Triton server 
```bash 
docker run --gpus "device=2" --rm -p8330:8000 -p8331:8001 -p8332:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:21.12-py3 tritonserver --model-repository=/models --strict-model-config false --log-verbose 1

...
+------------+---------+--------+
| Model      | Version | Status |
+------------+---------+--------+
|magface_onnx| 1       | READY  |
|magface_trt | 1       | READY  |
+------------+---------+--------+
I0714 00:37:55.265177 1 grpc_server.cc:4062] Started GRPCInferenceService at 0.0.0.0:8001
I0714 00:37:55.269588 1 http_server.cc:2887] Started HTTPService at 0.0.0.0:8000
I0714 00:37:55.312507 1 http_server.cc:2906] Started Metrics Service at 0.0.0.0:8002
```

Python client 
```bash 
python3 client.py dummy --model magface_trt --width 112 --height 112
```

### Benchmark

```bash 
docker run -it --ipc=host --net=host nvcr.io/nvidia/tritonserver:21.12-py3-sdk /bin/bash\
cd install/bin
perf_analyzer -m magface_trt --percentile=95 --concurrency-range 1:4 -u localhost:8330 --shape input:1,3,112,112 --measurement-interval 10000
...
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 1, throughput: 560.4 infer/sec, latency 2095 usec
Concurrency: 2, throughput: 1242.8 infer/sec, latency 2007 usec
Concurrency: 3, throughput: 1093.2 infer/sec, latency 2619 usec
Concurrency: 4, throughput: 913.8 infer/sec, latency 3766 usec
```
