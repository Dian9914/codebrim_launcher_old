#!/bin/bash

python ./download_dataset.py <URL>

jar xvf codebrim.zip

python ./codebrim2coco.py ./original_dataset/ -o ./codebrim_coco/