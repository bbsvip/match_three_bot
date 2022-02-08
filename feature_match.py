""" Created by MrBBS """
# 2/7/2022
# -*-encoding:utf-8-*-

from pathlib import Path
import cv2
import numpy as np
import time

OBJECT_WIDTH_LIMITER = 300
sift = cv2.SIFT_create()

feature_detector = cv2.KAZE_create()
bf = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=2)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)


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


def get_features(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # kp, des = sift.detectAndCompute(img, None)
    kp, des = feature_detector.detectAndCompute(img, None)
    return kp, des


def get_match(des1, des2, threshold=0.5):
    # kp1, des1 = get_features(img1)
    # kp2, des2 = get_features(img2)
    # matches = flann.knnMatch(des1, des2, k=2)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [[m] for m, n in matches if m.distance < threshold * n.distance]
    return good_matches, kp1, des1


template = {
    'red': [],
    'blue': [],
    'green': [],
    'pink': [],
    'violet': [],
    'yellow': []
}
color = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'pink': (255, 0, 255),
    'violet': (255, 0, 255),
    'yellow': (0, 255, 255)
}
for p in Path('template').rglob('*.[jp][pn]*'):
    name = p.name.split('.')[0].split('_')[0]
    tem = cv2.imread(p.as_posix())
    _, des = get_features(tem)
    template[name].append(des)
img = cv2.imread('image (1).jpg')
copy_img = img.copy()

white_mask = cv2.inRange(img, (180, 180, 180), (255, 255, 255))
white_mask = white_mask.astype(float) / 255
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, np.ones((1, 10), np.uint8))

h, s, v = cv2.split(img)
_, mask = cv2.threshold(v, 0, 1, cv2.THRESH_OTSU)

mask[white_mask == 1] = 0
n, conComp, stats, centroids = cv2.connectedComponentsWithStats(mask)

kp1, des1 = get_features(img)

start_time = time.time()
for k, v in template.items():
    for des in v:
        good, kp1, kp2 = get_match(des1, des)
        good_keypoints = [kp1[match[0].queryIdx].pt for match in good]

        cc_filtered = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        for kp in good_keypoints:
            ccNumber = conComp[int(kp[1]), int(kp[0])]

            m = np.zeros((img.shape[0], img.shape[1]), np.uint8)
            if ccNumber != 0:
                if int(kp[0]) - OBJECT_WIDTH_LIMITER < 0:
                    left_limit = 0
                else:
                    left_limit = int(kp[0]) - OBJECT_WIDTH_LIMITER

                if int(kp[0]) + OBJECT_WIDTH_LIMITER > img.shape[0]:
                    right_limit = img.shape[0]
                else:
                    right_limit = int(kp[0]) + OBJECT_WIDTH_LIMITER

                m[conComp == ccNumber] = 1
                m[:, right_limit:] = 0
                m[:, :left_limit] = 0
                cc_filtered[m == 1] = ccNumber

        n, _, stats, _ = cv2.connectedComponentsWithStats(cc_filtered)
        boxes = []
        for ccNumber in range(n):
            if ccNumber != 0:
                tl = (stats[ccNumber, cv2.CC_STAT_LEFT], stats[ccNumber, cv2.CC_STAT_TOP])
                br = (
                    stats[ccNumber, cv2.CC_STAT_LEFT] + stats[ccNumber, cv2.CC_STAT_WIDTH],
                    stats[ccNumber, cv2.CC_STAT_TOP] + stats[ccNumber, cv2.CC_STAT_HEIGHT],
                )
                boxes.append(tl + br)
        boxes = non_max_suppression(np.array(boxes))
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(copy_img, (x1, y1), (x2, y2), color[k], 5)
cv2.namedWindow('aa', cv2.WINDOW_NORMAL)
cv2.imshow('aa', copy_img)
print(time.time() - start_time)
cv2.waitKey()
