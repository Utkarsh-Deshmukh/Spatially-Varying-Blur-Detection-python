import cv2
import numpy as np
from BlurDetector import BlurDetector

if __name__ == '__main__':
    img = cv2.imread('images/motion0138.jpg', 0)
    blur_detector = BlurDetector()

    map = blur_detector.detectBlur(img)

    cv2.imshow('a', img)
    cv2.imshow('b', map/np.max(map))
    cv2.waitKey(0)