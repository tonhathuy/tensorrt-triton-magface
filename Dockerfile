# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/tensorrt:21.12-py3

# Install linux packages
RUN apt update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx

# Install python dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install gsutil gdown 

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

# Copy weights
# RUN gdown --id 1r7R8n27_0RA5l7fJX0-5O5pdYWw1duiS && \
#     unzip -r /usr/src/app/weights.zip && \

RUN chmod +x ./convert.sh
# Convert pytorch to onnx to TensorRT 
ENTRYPOINT ["/bin/bash", "./convert.sh"]
