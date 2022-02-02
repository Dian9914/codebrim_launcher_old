#!/bin/bash

while read url; do
    wget $url -O ./codebrim.zip
done < urls.txt

jar xvf codebrim.zip

python ./codebrim2coco.py ./original_dataset/ -o ./codebrim_coco/
