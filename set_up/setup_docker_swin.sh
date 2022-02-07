#!/bin/bash

cd /workspace/apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd /workspace/mmdetection-swin
mkdir data
mv /workspace/data/codebrim_coco /workspace/mmdetection-swin/data
