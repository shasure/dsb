#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author : zsy
Date : 2018/02/02"""

import argparse
import os

import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm

"""将stage1_train中的input和masksmerged存储到html中，方便查看"""

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="/datasets/dsb",
                    help="e.g. /datasets/dsb/")
parser.add_argument("--subdir", type=str, default="stage1_train",
                    help="e.g. stage1_train")

a = parser.parse_args()


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>images</th><th>masks</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["images", "masks"]:
            index.write("<td><img src='%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path


def gen_filesets(step=None):
    dirnames = next(os.walk(os.path.join(a.output_dir, a.subdir)))[1]
    paths = [os.path.join(a.subdir, id_, 'images', id_ + '.png') for id_ in dirnames]
    mask_paths = [os.path.join(a.subdir, id_, 'masksmerged', id_ + '.png') for id_ in dirnames]

    filesets = []
    for i, in_path in enumerate(paths):
        fileset = {"name": dirnames[i], "step": step}
        fileset['images'] = in_path
        fileset['masks'] = mask_paths[i]

        filesets.append(fileset)
    return filesets


if __name__ == '__main__':
    filesets = gen_filesets()
    index_path = append_index(filesets)
    print('store train images to html in %s' % index_path)
