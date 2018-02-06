#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author : zsy
Date : 2018/02/01"""
import argparse
import os

import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm

"""生成fake mask image for test"""

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/datasets/dsb/stage1_test",
                    help="e.g. /datasets/dsb/stage1_test")

flag = parser.parse_args()
print(flag)

if __name__ == '__main__':

    dirslist = next(os.walk(flag.data_dir))[1]
    print(len(dirslist))
    print(dirslist)
    print('start merge masks')

    for n, id_ in tqdm(enumerate(dirslist), total=len(dirslist)):  # 处理每个training image
        path = os.path.join(flag.data_dir, id_)  # dir path
        if not os.path.exists(os.path.join(path, 'masksmerged')):  # 创建masksmerged子文件夹
            os.mkdir(os.path.join(path, 'masksmerged'))
        else:
            raise FileExistsError("masksmerged dir already exists")
        img_file = imread(os.path.join(path, 'images', id_ + '.png'))
        fake_mask = np.zeros((img_file.shape[0], img_file.shape[1]), dtype=np.uint8)  # 处理training image mask
        # save merged mask image
        imsave(os.path.join(path, 'masksmerged', id_ + '.png'), fake_mask)
