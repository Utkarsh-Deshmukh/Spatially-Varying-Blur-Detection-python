# Spatially-Varying-Blur-Detection-python
python implementation of the paper "Spatially-Varying Blur Detection Based on Multiscale Fused and Sorted Transform Coefficients of Gradient Magnitudes" - cvpr 2017

# Installation and Running an example:
## method 1 - use the library:
`pip install blur_detector`

```
import blur_detector
import cv2
if __name__ == '__main__':
    img = cv2.imread('image_name', 0)
    blur_map = blur_detector.detectBlur(img, downsampling_factor=4, num_scales=4, scale_start=2, num_iterations_RF_filter=3)

    cv2.imshow('ori_img', img)
    cv2.imshow('blur_map', blur_map)
    cv2.waitKey(0)
```

# Results
 ### Detecting Depth of field:
![image](https://user-images.githubusercontent.com/13918778/119441249-aa3dc780-bcda-11eb-911b-432266dfa92c.png)
--------------------------------------------------------------------------------------------------------------
 ### Detecting various types of blurs:
![image](https://user-images.githubusercontent.com/13918778/119441726-74e5a980-bcdb-11eb-8d55-55b3e2c5f7be.png)
![image](https://user-images.githubusercontent.com/13918778/119441933-cee66f00-bcdb-11eb-907e-776ed1f47054.png)
![image](https://user-images.githubusercontent.com/13918778/119442075-09e8a280-bcdc-11eb-826a-cf8277f3c7cc.png)
--------------------------------------------------------------------------------------------------------------

# Acknowledgements
This algorithm is based on the paper: `Spatially-Varying Blur Detection Based on Multiscale Fused and SortedTransform Coefficients of Gradient Magnitudes`
The paper can be found here: https://arxiv.org/pdf/1703.07478.pdf

The author would like to thank Dr. Alireza Golestaneh (This code is a python implementation of his work. His original work is an Matlab and can be found here: https://github.com/isalirezag/HiFST)
