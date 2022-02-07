#!/bin/bash

cd ./data

wget -i urls.txt $url -O ./codebrim.zip
jar xvf codebrim.zip
python ./codebrim2coco.py ./original_dataset/ -o ./codebrim_coco/

rm ./codebrim.zip
rm -r ./original_dataset 
rm -r ./__MACOSX