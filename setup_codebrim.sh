#!/bin/bash

while read url; do
    wget $url
done < urls.txt

jar xvf codebrim.zip

python ./codebrim2coco.py ./original_dataset/ -o ./codebrim_coco/
