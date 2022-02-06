 
import cv2
 
onnx_model_path = "./inference/models/magface_iresnet18_casia_dp.onnx"
sample_image = "./inference/toy_imgs/0.jpg"
 
net =  cv2.dnn.readNetFromONNX(onnx_model_path) 
image = cv2.imread(sample_image)
blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (112, 112),(0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
preds = net.forward()
print(preds)
print(preds.shape)
