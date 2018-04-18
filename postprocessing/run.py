import os
import PIL
from PIL import Image
import numpy as np
from skimage.morphology import label
import sys

"""对mask-rcnn产生的mask进行处理"""

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x):
    lab_img = label(x)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


def IOU(mask1, mask2):
    m = np.logical_and(mask1, mask2)
    m2 = np.logical_or(mask1, mask2)
    return np.sum(m) / np.sum(m2)


def merge(mask1, mask2):
    return np.logical_or(mask1, mask2)


def merge_all(masks, t):
    m = list(masks)
    while True:
        temp = []
        del_nums = set([])
        for i in range(len(m) - 1):
            for j in range(i + 1, len(m)):
                mask1 = m[i]
                mask2 = m[j]
                if IOU(mask1, mask2) > t:
                    print(i, j)
                    temp.append(merge(mask1, mask2))
                    del_nums.add(i)
                    del_nums.add(j)
        for i in range(len(m)):
            if i not in del_nums:
                temp.append(m[i])
        m = list(temp)
        print(len(del_nums), len(m))
        if len(del_nums) == 0:
            break
    return m


def run(r, cutoff):
    image_ids = []
    base = './' + r + '/result/'
    image_ids = os.listdir(base)
    # if idx.split('.')[1]=='png' and len(idx.split('.'))==2:
    #     image_ids.append(idx.split('.')[0])

    result = {}
    for idx in image_ids:
        print(idx)
        result[idx] = []
        masks = []
        masks_b = []
        for file in os.listdir(base + '%s/' % idx):
            image = Image.open(base + idx + '/' + file)
            image = np.array(image) > cutoff
            masks.append(image)
        blank = np.zeros_like(masks[0])

        # for i in range(len(masks)-1):
        #     if i not in del_nums: 
        #         m = masks[i]
        #         for j in range(i,len(masks)):
        #             if IOU(m,masks[j])>0.7 and j not in del_nums:
        #                 m = merge(m,masks[j])
        #                 del_nums.append(j)
        #         masks_b.append(m)
        # masks_b = merge_all(masks,0.5)
        masks_b = masks
        for image in masks_b:
            image = np.logical_and(image, np.logical_not(blank))
            blank = np.logical_or(image, blank)
            result[idx].append(rle_encoding(image))

        # image = Image.open(base+idx+'.png')
        # image = np.array(image)
        # print(blank.any())
        # for rle in prob_to_rles(blank):
        #     result[idx].append(rle)

    with open('./%s_%d.csv' % (r, cutoff), 'w', encoding='utf-8') as file:
        file.write('ImageId,EncodedPixels\n')
        for idx in result:
            for code in result[idx]:
                if len(code) > 0:
                    file.write(idx)
                    file.write(',')
                    for c in code:
                        file.write('%d ' % c)
                    file.write('\n')


def test():
    base = './' + 'r9' + '/result/'
    image_ids = os.listdir(base)
    for idx in image_ids:
        mask = os.listdir(base + idx)
        for i in range(len(mask) - 1):
            m1 = Image.open(base + idx + '/' + mask[i])
            m1 = np.array(m1) > 100
            for j in range(i, len(mask)):
                m2 = Image.open(base + idx + '/' + mask[j])
                m2 = np.array(m2) > 100
                if np.logical_and(m1, m2).any():
                    print(base + idx + '/' + mask[j], base + idx + '/' + mask[i])


if __name__ == '__main__':
    # test()
    run(sys.argv[1], int(sys.argv[2]))
