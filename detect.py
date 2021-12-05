import cv2
import numpy as np

red = [np.asarray([0, 0, 116]), np.asarray([34, 72, 255])]
blue = [np.asarray([81, 81, 0]), np.asarray([255, 169, 90])]
green = [np.asarray([0, 146, 0]), np.asarray([127, 255, 155])]
pink = [np.asarray([162, 0, 209]), np.asarray([255, 151, 255])]
violet = [np.asarray([151, 74, 139]), np.asarray([255, 157, 199])]
yellow = [np.asarray([144, 232, 239]), np.asarray([172, 255, 255])]
color_range = {
    'red   ': red,
    'blue  ': blue,
    'green ': green,
    'pink  ': pink,
    'violet': violet,
    'yellow': yellow
}


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
    for c in cnt:
        x, y, w, h = cv2.boundingRect(c)
        if w > 50 and h > 50:
            yield x + w // 2, y + h // 2


def get_board(img, rows, columns):
    img = img[915: img.shape[0] - 40, :, :]
    h, w = img.shape[:2]
    coords = []
    for k in color_range.keys():
        coord = get_coords(k, img)
        for x, y in coord:
            coords.append([x, y, k])
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
