import os
import os.path as osp

from natsort import natsorted, ns

for split in ['train', 'val']:
        devkit_path = './data/codebrim'
        dataset_name = 'codebrim' + '_' + split
        print(f'processing {dataset_name} ...')
        filelist = next(os.walk(osp.join(devkit_path, f'{split}/annotations/')), (None, None, []))[2] 
        filelist = natsorted(filelist, key=lambda y: y.lower())
        filelist = [ name.replace('.xml','') for name in filelist ]
        img_names = filelist
        for name in img_names:
            os.rename(osp.join(devkit_path, f'{split}/images/{name}.jpg'),osp.join(devkit_path, f'{split}/ann_imgs/{name}.jpg'))
