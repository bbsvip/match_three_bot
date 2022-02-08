
import cv2
import numpy as np


# trackbar callback fucntion does nothing but required for trackbar
def nothing(x):
    pass


# create a seperate window named 'controls' for trackbar
cv2.namedWindow('controls')
# create trackbar in 'controls' window with name 'r''
cv2.createTrackbar('R low', 'controls', 0, 255, nothing)
cv2.createTrackbar('G low', 'controls', 0, 255, nothing)
cv2.createTrackbar('B low', 'controls', 0, 255, nothing)
cv2.createTrackbar('R upper', 'controls', 0, 255, nothing)
cv2.createTrackbar('G upper', 'controls', 0, 255, nothing)
cv2.createTrackbar('B upper', 'controls', 0, 255, nothing)

img = cv2.imread('image (1).jpg')
# img = img[915: img.shape[0] - 40, :, :]
h, w = img.shape[:2]
# img = cv2.resize(img, (w // 2, h // 2))
cv2.namedWindow('ori', cv2.WINDOW_NORMAL)
cv2.namedWindow('cc', cv2.WINDOW_NORMAL)
cv2.imshow('ori', img)
# create a while loop act as refresh for the view
while (1):
    r_low = int(cv2.getTrackbarPos('R low', 'controls'))
    g_low = int(cv2.getTrackbarPos('G low', 'controls'))
    b_low = int(cv2.getTrackbarPos('B low', 'controls'))

    r_upper = int(cv2.getTrackbarPos('R upper', 'controls'))
    g_upper = int(cv2.getTrackbarPos('G upper', 'controls'))
    b_upper = int(cv2.getTrackbarPos('B upper', 'controls'))

    hsv_color1 = np.asarray([b_low, g_low, r_low])
    hsv_color2 = np.asarray([b_upper, g_upper, r_upper])

    mask = cv2.inRange(img, hsv_color1, hsv_color2)
    cv2.imshow('cc', mask)

    # waitfor the user to press escape and break the while loop 
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

# destroys all window
cv2.destroyAllWindows()
