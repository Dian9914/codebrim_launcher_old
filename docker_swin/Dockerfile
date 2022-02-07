ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ARG MMCV="1.3.17"
ARG MMDET="2.20.0"

ENV PYTHONUNBUFFERED TRUE

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    g++ \
    openjdk-8-jre \
    wget \
    unzip \
    # MMDet Requirements
    ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/conda/bin:$PATH"
RUN export FORCE_CUDA=1

# TORCHSEVER
RUN pip install torchserve torch-model-archiver

# MMLAB
ARG PYTORCH
ARG CUDA
RUN ["/bin/bash", "-c", "pip install mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]
RUN pip install mmdet==${MMDET}

# CODE FOR DATASET DOWNLOAD
#mkdir ./data && \
#cd ./data && \
#wget https://uses0-my.sharepoint.com/:u:/g/personal/edusotcas_alum_us_es/EWM7b1KwfKFOlCtP82_CCb8BKmKJR5Ve9IH2i62GnuBXZQ?download=1 -O ./codebrim.zip && \
#unzip codebrim.zip && \
#rm ./codebrim.zip && \
#cd .. && \
#   

RUN cd ~

# MMDETECTION
RUN git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection.git -o mmdetection-swin && \
    cd mmdetection-swin && \
    pip install -r requirements/build.txt && \
    pip install -v -e . && \
    cd ..

# APEX
RUN git clone https://github.com/NVIDIA/apex 

    
