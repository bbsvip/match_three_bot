import cv2
import numpy as np

red = [np.asarray([0, 0, 116]), np.asarray([34, 72, 255])]
blue = [np.asarray([81, 81, 0]), np.asarray([255, 169, 90])]
green = [np.asarray([0, 146, 0]), np.asarray([127, 255, 155])]
pink = [np.asarray([162, 0, 209]), np.asarray([255, 151, 255])]
violet = [np.asarray([151, 74, 139]), np.asarray([255, 157, 199])]
yellow = [np.asarray([144, 232, 239]), np.asarray([172, 255, 255])]
power = [np.array([0, 255, 0]), np.array([125, 255, 255])]
color_range = {
    'red   ': red,
    'blue  ': blue,
    'green ': green,
    'pink  ': pink,
    'violet': violet,
    'yellow': yellow,
    'power': power
}

colors = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'pink': (255, 0, 255),
    'violet': (255, 0, 255),
    'yellow': (0, 255, 255),
    'power': (255, 255, 255)
}


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


def get_coords(color_code, img):
    coord = []
    # img = img.copy()
    color = color_range[color_code]
    mask = cv2.inRange(img, color[0], color[1])
    kernel = np.ones((5, 5), np.uint8)
    if color_code != 'yellow':
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=5)
    cnt, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bboxs = []
    for c in cnt:
        x, y, w, h = cv2.boundingRect(c)
        bboxs.append([x, y, x + w, y + h])
    bboxs = non_max_suppression(np.array(bboxs))
    for x1, y1, x2, y2 in bboxs:
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[color_code.strip()], 2)
        if x2 - x1 > 50 and y2 - y1 > 50:
            yield x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2


def get_board(img, rows, columns):
    # img = img[915: img.shape[0] - 40, :, :]
    h, w = img.shape[:2]
    coords = []
    for k in color_range.keys():
        coord = get_coords(k, img)
        for x, y in coord:
            coords.append([x, y, k])
    cv2.namedWindow('cc', cv2.WINDOW_NORMAL)
    cv2.imshow('cc', img)
    cv2.waitKey()
    stepx = w // columns
    stepy = h // rows
    x_tile = 0
    y_tile = 0
    board = [[1] * columns for _ in range(rows)]
    coord_board = [[1] * columns for _ in range(rows)]
    for x in range(rows):
        for y in range(columns):
            for cx, cy, ck in coords:
                if cx in range(x_tile, x_tile + stepy) and cy in range(y_tile, y_tile + stepx):
                    board[x][y] = ck
                    coord_board[x][y] = [cx, cy]
                    break
            x_tile += stepy
        x_tile = 0
        y_tile += stepx
    for br in board:
        for b in br:
            if b == 1:
                return None, None
    return board, coord_board
