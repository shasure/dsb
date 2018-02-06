#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author : zsy
Date : 2018/01/30"""
import argparse
import os

import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm

"""将stage1_train中所有subdir中的masks整合到一张图片中，存储到masksmerged子文件夹中"""

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/datasets/dsb/stage1_train",
                    help="e.g. /datasets/dsb/stage1_train")

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
        mask = None
        for mask_file in next(os.walk(os.path.join(path, 'masks')))[2]:
            mask_ = imread(os.path.join(path, 'masks', mask_file))
            if mask is None:
                mask = np.zeros((mask_.shape[0], mask_.shape[1]), dtype=np.uint8)  # 处理training image mask
            mask = np.maximum(mask, mask_)  # 将mask叠加到一起，白色为255，黑色为0
        # save merged mask image
        imsave(os.path.join(path, 'masksmerged', id_ + '.png'), mask)
