# Efficient-Spatially-Varying-Blur-Detection-python
python implementation of the paper "***Spatially-Varying Blur Detection Based on Multiscale Fused and Sorted Transform Coefficients of Gradient Magnitudes" - cvpr 2017***

***NOTE: This project outputs regions in an image which are sharp and blurry. In order to perform "OUT-OF-FOCUS" blur estimation, please refer to this repo: https://github.com/Utkarsh-Deshmukh/Blurry-Image-Detector***

# Brief Theory:
- The repo is a python implementation of the paper which can be found here
[link to the paper](https://arxiv.org/pdf/1703.07478.pdf)
- The discrete cosine transform is used to convert the image from spatial domain to frequency domain.
- The DCT coefficients are divided into 'low', 'medium' and 'high' frequency bands, out of which only the high frequencies are used.
- At a particular location, the high frequency DCT coefficients are extracted at various resolutions. All these selected coefficients are combined together and sorted to form the `multiscale-fused and sorted high-frequency transform coefficients`
- these coefficients show a visual difference for blurry vs sharp patches in the image (refer to fig 2 in the paper)
- each of these coefficients results in the generation of a "filtered layer" for the image. The first layer is the smallest DCT high frequency coefficient while the last layer is the highest DCT high frequency coefficient at various scales. (refer to secion 2.1 in the paper)
- The union of a subset of these layers is selected. This is then sent through a max pooling to retain the highest activation from the set of layers
- This resultant map is then sent for post processing which involves computing the local entropy and smoothing this local entropy map using a edge retaining smoothing filter such as `Domain transform recursive edge-preserving filter`. A guided filter can also be used here.
- The *Domain transform recursive edge-preserving filter* was proposed in 2011 and can be found here:
[link to the paper](https://www.inf.ufrgs.br/~eslgastal/DomainTransform/Gastal_Oliveira_SIGGRAPH2011_Domain_Transform.pdf)

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
--------------------------------------------------------------------------------------------------------------
# Results
--------------------------------------------------------------------------------------------------------------
### Detecting Depth of field:
![image](https://user-images.githubusercontent.com/13918778/119441249-aa3dc780-bcda-11eb-911b-432266dfa92c.png)
--------------------------------------------------------------------------------------------------------------
 ### Detecting various types of blurs:
![image](https://user-images.githubusercontent.com/13918778/119441726-74e5a980-bcdb-11eb-8d55-55b3e2c5f7be.png)
![image](https://user-images.githubusercontent.com/13918778/119441933-cee66f00-bcdb-11eb-907e-776ed1f47054.png)
![image](https://user-images.githubusercontent.com/13918778/119442075-09e8a280-bcdc-11eb-826a-cf8277f3c7cc.png)
--------------------------------------------------------------------------------------------------------------
# Algorithm overview:
![image](https://user-images.githubusercontent.com/13918778/119443637-c7749500-bcde-11eb-9b71-c16210e39910.png)
--------------------------------------------------------------------------------------------------------------
# Development/Testing Environment 
This repository uses `poetry` as a package manager

```
Make sure your python version is >= 3.7
python --version

# Install poetry if you don't have it
python -m pip install poetry

# Install dependencies in a virtual environment
poetry install 
(This will install all locked dependencies from poetry.lock file)

# Activate virtual environment
poetry shell (Now you have a virtual environment with all dependencies)
Alternatively, to avoid creating a new shell, you can manually activate the virtual environment by running `source {path_to_venv}/bin/activate`
To get the path to your virtual environment run `poetry env info --path`. You can also combine these into a nice one-liner, `source $(poetry env info --path)/bin/activate`. To deactivate this virtual environment simply use `deactivate`.

# Using poetry run
To run your script simply use poetry run python your_script.py. This will automatically run your script in above created shell   

# Specifying dependencies
If you want to add dependencies to this project, you can specify them in the `tool.poetry.dependencies` section.

For example;

[tool.poetry.dependencies]
pendulum = "^1.4"

As you can see, it takes a mapping of package names and version constraints.
Poetry uses this information to search for the right set of files in package "repositories" that you register in the tool.poetry.repositories section, or on PyPI by default. 
Also, instead of modifying the pyproject.toml file by hand, you can use the add command.

poetry add pendulum

It will automatically find a suitable version constraint and install the package and subdependencies.

```
--------------------------------------------------------------------------------------------------------

# Publish to Pypi
Before you can actually publish your library, you will need to package it.

```
poetry build
```

This command will package your library in two different formats: `sdist` which is the source format, and `wheel` which is a `compiled` package.
Once that is done you are ready to publish your library.

Poetry will publish to PyPI by default. Anything that is published to PyPI is available automatically through Poetry.

To share `blur_detection` library with the Python community, we would publish on PyPI as well. Doing so is really easy.

```
poetry publish
```

This will package and publish the library to PyPI, at the condition that you are a registered user and you have configured your credentials properly.

---------------------------------------------------------------------------------------------------

# Acknowledgements
This algorithm is based on the paper: `Spatially-Varying Blur Detection Based on Multiscale Fused and SortedTransform Coefficients of Gradient Magnitudes`
The paper can be found here: https://arxiv.org/pdf/1703.07478.pdf

The author would like to thank Dr. Alireza Golestaneh (This code is a python implementation of his work. His original work is an Matlab and can be found here: https://github.com/isalirezag/HiFST)
