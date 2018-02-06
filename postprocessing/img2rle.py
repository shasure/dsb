#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author : zsy
Date : 2018/02/01"""
import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.measure import label
from skimage.transform import resize

parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, default=0.5,
                    help='threshold of masks, masks are float type images between [0,1)')
parser.add_argument('--test_dir', type=str, default="/home/zsy/train_dir/dsb/dsb_test/images",
                    help="e.g. /home/zsy/train_dir/dsb/dsb_test/images")
parser.add_argument('--dbtest_dir', type=str, default="/home/zsy/datasets/dsb/stage1_test",
                    help="e.g. /home/zsy/datasets/dsb/stage1_test")
parser.add_argument('--dbtrain_dir', type=str, default="/home/zsy/datasets/dsb/stage1_train",
                    help="e.g. /home/zsy/datasets/dsb/stage1_train")
a = parser.parse_args()


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


def get_test_img_size(test_dir, test_ids):
    sizes_test = []

    print('Getting test images size... ')
    for n, id_ in enumerate(test_ids):
        path = os.path.join(test_dir, id_)
        img = imread(path + '/images/' + id_ + '.png')[:, :, :3]
        sizes_test.append([img.shape[0], img.shape[1]])
    return sizes_test


def load_predicted_test_img(test_dir, test_ids):
    preds_test = []
    for n, id_ in enumerate(test_ids):
        preds_test.append(imread(os.path.join(test_dir, id_ + '-outputs.png')))
    return preds_test


if __name__ == '__main__':

    # first get test image original size
    test_ids = next(os.walk(a.dbtest_dir))[1]
    sizes_test = get_test_img_size(a.dbtest_dir, test_ids)

    # load predicted test img
    preds_test = load_predicted_test_img(a.test_dir, test_ids)

    # Create list of upsampled test masks
    preds_test_upsampled = []
    for i in range(len(preds_test)):
        preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                           (sizes_test[i][0], sizes_test[i][1]),
                                           mode='constant', preserve_range=True))

    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(preds_test_upsampled[n], cutoff=int(a.threshold * 255)))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))

    # ... and then finally create our submission!

    # In[ ]:

    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('pix2pix%s.csv' % datetime.now().strftime('_%d_%m_%Y_%H_%M'), index=False)
