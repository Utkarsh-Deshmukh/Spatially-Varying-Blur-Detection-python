import cv2
import numpy as np
from BlurDetector import BlurDetector

if __name__ == '__main__':
    img = cv2.imread('images/motion0138.jpg', 0)
    blur_detector = BlurDetector(downsampling_factor=4, num_scales=3, scale_start=2, entropy_filt_kernel_sze=7, sigma_s_RF_filter=15, sigma_r_RF_filter=0.25, num_iterations_RF_filter=3, show_progress=True)

    map = blur_detector.detectBlur(img)

    cv2.imshow('a', img)
    cv2.imshow('b', map/np.max(map))
    cv2.waitKey(0)