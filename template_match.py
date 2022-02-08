""" Created by MrBBS """
# 2/8/2022
# -*-encoding:utf-8-*-

import cv2
import numpy as np
from pathlib import Path
import time


def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int")


img_rgb = cv2.imread('image (1).jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = {
    'first': [],
    'second': [],
    'green': [],
    'pink': [],
    'violet': [],
    'yellow': []
}

color = {
    'first': (0, 0, 255),
    'second': (255, 255, 0),
    'green': (0, 255, 0),
    'pink': (255, 0, 255),
    'violet': (255, 125, 255),
    'yellow': (0, 255, 255)
}

for p in Path('template_line').rglob('*.[jp][pn]*'):
    name = p.name.split('.')[0].split('_')[0]
    tem = cv2.imread(p.as_posix(), 0)
    template[name].append(tem)
# template = cv2.imread('template/red_scale.png', 0)
# w, h = template.shape[::-1]
start_time = time.time()
for k, v in template.items():
    boxes = []
    # cv2.namedWindow(k, cv2.WINDOW_NORMAL)
    # copy_image = img_rgb.copy()
    for tem in v:
        h, w = tem.shape
        res = cv2.matchTemplate(img_gray, tem, cv2.TM_CCOEFF_NORMED)
        threshold = 0.9
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            boxes.append(pt + (pt[0] + w, pt[1] + h))
        for x1, y1, x2, y2 in non_max_suppression(np.array(boxes)):
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color[k], 2)
    # cv2.imshow(k, copy_image)
print(time.time() - start_time)
cv2.namedWindow('res', cv2.WINDOW_NORMAL)
cv2.imshow('res', img_rgb)
cv2.waitKey()
