#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author : zsy
Date : 2018/02/01"""
import argparse
import csv

import os

parser = argparse.ArgumentParser()
parser.add_argument('--csvfile', type=str, default='pix2pix_01_02_2018_20_42.csv',help='name of csv file')
parser.add_argument('--threshold', type=int, default=6, help='threshold of pixels')
a = parser.parse_args()


if __name__ == '__main__':
    rows = []
    name_postfix = a.csvfile.split('.')
    out_name = name_postfix[0] + '_truncated.' + name_postfix[1]
    with open(a.csvfile) as csvfile:
        with open(out_name, 'w') as outfile:
            reader = csv.reader(csvfile)
            writer = csv.writer(outfile)
            fieldnames = reader.__next__()
            writer.writerow(fieldnames)
            for row in reader:
                if len(row[1].split()) /2 >= a.threshold:
                    writer.writerow(row)

