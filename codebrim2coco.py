# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import walk, rename, remove
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np

import mmcv
from PIL import Image
import json
import numpy as np

import cv2
from cv2 import ROTATE_90_CLOCKWISE

import random

def parse_xml(args):
    ann_path, img_path, img_name = args
    
    tree = ET.parse(ann_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    bboxes = []
    labels = []
    bboxes_ignore = []
    labels_ignore = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        label=[]
        difficult = int(obj.find('difficult').text)
        bnd_box = obj.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        defect = obj.find('Defect')
        if int(defect.find('Background').text): label = label + [1]
        if int(defect.find('Crack').text): label = label + [2]
        if int(defect.find('Spallation').text): label = label + [3]
        if int(defect.find('Efflorescence').text): label = label + [4]
        if int(defect.find('ExposedBars').text): label = label + [5]
        if int(defect.find('CorrosionStain').text): label = label + [6]
        
        for lbl in label:
            if difficult:
                bboxes_ignore.append(bbox)
                labels_ignore.append(lbl)
            else:
                bboxes.append(bbox)
                labels.append(lbl)
              
    if not bboxes:
        bboxes = np.zeros((0, 4))
        labels = np.zeros((0, ))
    else:
        bboxes = np.array(bboxes, ndmin=2) - 1
        labels = np.array(labels)
    if not bboxes_ignore:
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0, ))
    else:
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
        labels_ignore = np.array(labels_ignore)
    annotation = {
        'filename': img_name,
        'width': w,
        'height': h,
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
            'bboxes_ignore': bboxes_ignore.astype(np.float32),
            'labels_ignore': labels_ignore.astype(np.int64)
        }
    }
    return annotation


def cvt_annotations(devkit_path, split, out_dir, out_file):
    # obtenemos las anotaciones del split correspondiente
    ann_names = next(walk(osp.join(devkit_path, f'{split}/annotations/')), (None, None, []))[2] 
    ann_paths = [
        osp.join(devkit_path, f'{split}/annotations/{ann_name}')
        for ann_name in ann_names
    ]
    img_names = next(walk(osp.join(devkit_path, f'{split}/images/')), (None, None, []))[2] 
    img_o_paths = [
        osp.join(devkit_path,f'{split}/images/{img_name}') for img_name in img_names
    ]
    annotations = []
    part_annotations = mmcv.track_progress(parse_xml,
                                            list(zip(ann_paths, img_o_paths, img_names)))
    annotations.extend(part_annotations)
    print('Done!')
    print(f'Moving {split} imgs to new folder...')
    img_d_paths = [
        osp.join(out_dir,f'{split}/{img_name}') for img_name in img_names
    ]
    for img_o, img_d in zip(img_o_paths,img_d_paths):
        rename(img_o, img_d)
    if out_file.endswith('json'):
        annotations = cvt_to_coco_json(annotations)
    mmcv.dump(annotations, osp.join(out_dir,'annotations/'+out_file))
    return annotations


def cvt_to_coco_json(annotations):
    image_id = 0
    annotation_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    def addAnnItem(annotation_id, image_id, category_id, bbox, difficult_flag):
        annotation_item = dict()
        annotation_item['segmentation'] = []

        seg = []
        # bbox[] is x1,y1,x2,y2
        # left_top
        seg.append(int(bbox[0]))
        seg.append(int(bbox[1]))
        # left_bottom
        seg.append(int(bbox[0]))
        seg.append(int(bbox[3]))
        # right_bottom
        seg.append(int(bbox[2]))
        seg.append(int(bbox[3]))
        # right_top
        seg.append(int(bbox[2]))
        seg.append(int(bbox[1]))

        annotation_item['segmentation'].append(seg)

        xywh = np.array(
            [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        annotation_item['area'] = int(xywh[2] * xywh[3])
        if difficult_flag == 1:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 1
        else:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 0
        annotation_item['image_id'] = int(image_id)
        annotation_item['bbox'] = xywh.astype(int).tolist()
        annotation_item['category_id'] = int(category_id)
        annotation_item['id'] = int(annotation_id)
        coco['annotations'].append(annotation_item)
        return annotation_id + 1

    codebrim_clases = ['Background', 'Crack', 'Spallation', 'Efflorescence', 'ExposedBars', 'CorrosionStain']
    for category_id, name in enumerate(codebrim_clases):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id)+1
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    for ann_dict in annotations:
        file_name = ann_dict['filename']
        ann = ann_dict['ann']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['width'] = int(ann_dict['width'])
        image_item['height'] = int(ann_dict['height'])
        coco['images'].append(image_item)
        image_set.add(file_name)

        bboxes = ann['bboxes'][:, :4]
        labels = ann['labels']
        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            label = labels[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=0)

        bboxes_ignore = ann['bboxes_ignore'][:, :4]
        labels_ignore = ann['labels_ignore']
        for bbox_id in range(len(bboxes_ignore)):
            bbox = bboxes_ignore[bbox_id]
            label = labels_ignore[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=1)

        image_id += 1

    return coco


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert CODEBRIM annotations to mmdetection format')
    parser.add_argument('devkit_path', help='codebrim devkit path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--out-format',
        default='coco',
        choices=('pkl', 'coco'),
        help='output format, "coco" indicates coco annotation format')
    args = parser.parse_args()
    return args

def reorganice(devkit_path):
    # dividimos el dataset en imagenes para entrenamiento y para validacion y comprobamos que esten correctas
    print('Checking and moving files to new format... ')

    # borramos .DS_Store
    try: remove(osp.join(devkit_path,'annotations/.DS_Store'))
    except: pass
    
    # obtenemos los nombres de las anotaciones y sus paths
    ann_names = next(walk(osp.join(devkit_path, 'annotations/')), (None, None, []))[2] 
    ann_paths = [
        osp.join(devkit_path, f'annotations/{ann_name}')
        for ann_name in ann_names
    ]
    # los mezclamos de forma aleatoria para dirigir el 80% de las imagenes anotadas a entrenamiento
    # y el resto a validacion
    random.shuffle(ann_paths)
    n_train = round(len(ann_paths)*0.8)

    # definimos el path donde se guardaran las imagenes para entrenamiento
    split = 'train'
    mmcv.mkdir_or_exist(osp.join(devkit_path,f'{split}/images/'))
    mmcv.mkdir_or_exist(osp.join(devkit_path,f'{split}/annotations/'))
    cntr = 1
    for ann_o_path, ann in zip(ann_paths,ann_names):
        # obtenemos la direccion de la imagen correspondiente a cada anotacion, de este modo
        # solo tomamos imagenes anotadas
        img = ann.replace('.xml','.jpg')
        img_o_path = osp.join(devkit_path, f'images/{img}')

        # revisamos que las imagenes esten correctas
        if str(mmcv.imread(img_o_path).shape) != str(np.array(Image.open(img_o_path)).shape):
            print(f'The image {img_o_path} is wrong! Fixing it... ')
            image = cv2.imread(img_o_path)
            image = cv2.rotate(image, ROTATE_90_CLOCKWISE)
            image = cv2.imwrite(img_o_path, image)
        
        # guardamos las imagenes y notaciones en el directorio correspondiente
        img_d_path = osp.join(devkit_path, f'{split}/images/{img}')
        rename(img_o_path,img_d_path)
        ann_d_path = osp.join(devkit_path, f'{split}/annotations/{ann}')
        rename(ann_o_path,ann_d_path)

        # cuando hayamos repartido el 80%, cambiamos a validacion
        cntr = cntr + 1
        if cntr == n_train: 
            split = 'val'
            mmcv.mkdir_or_exist(osp.join(devkit_path,f'{split}/images/'))
            mmcv.mkdir_or_exist(osp.join(devkit_path,f'{split}/annotations/'))



def main():
    # leemos los argumentos y creamos las carpetas correspondientes
    args = parse_args()
    devkit_path = args.devkit_path
    out_dir = args.out_dir if args.out_dir else devkit_path
    mmcv.mkdir_or_exist(out_dir)
    mmcv.mkdir_or_exist(osp.join(out_dir,'annotations/'))

    # reorganizamos el dataset repartiendo entre imagenes para entrenamiento y para
    # validacion. Ademas, arreglamos las imagenes con errores de orientacion y 
    # purgamos aquellas no anotadas
    reorganice(devkit_path)

    # conversion de las anotaciones
    out_fmt = '.json'
    prefix='codebrim'
    for split in ['train', 'val']:
        mmcv.mkdir_or_exist(osp.join(out_dir,f'{split}'))
        dataset_name = prefix + '_' + split
        print(f'processing {dataset_name} ...')
        cvt_annotations(devkit_path, split, out_dir, dataset_name + out_fmt)
    print('Done!')


if __name__ == '__main__':
    main()
