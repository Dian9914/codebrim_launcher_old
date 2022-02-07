#!/bin/bash

cd /workspace/apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd /workspace/mmdetection
mkdir data
mv /workspace/docker_output/codebrim_coco /workspace/mmdetection/data