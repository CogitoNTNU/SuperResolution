# Super-Resolution Project Model Plan Spring 2022

## Overview
| Part | TODO |
| ---- | ---- |
| 1 | [Data Pre-Processing](#data-pre-processing) |
| 2 | [Model](#model)|
| 3 | [Utility](#utility) |
| 4 | [Metrics](#metrics) |
| 5 | [Running and Stop](#running-and-stop) |
| 6 | [Training](#training) |


---

## Data Pre-Processing
* Import dataset
* Split dataset into training and validation sections
* Scale all values in images to between 0 and 1
* Create test set
* Convert RGB to YUV, Split channels, downscale and return luminance
* Convert RGB to YUV, Split channels & return luminance from high res images
* Create training and validation datasets with high res and down scaled images 
  
## Model
* Activation Relu
* Kernel initializer orthogonal
* Pading same
* Variable input size
* Convolutional layers
* output is factor of input  

## Utility
* Plot and save output of model with zoom in area
* Convert image to its low res version
* Use model to upscale image and return in rgb

## Metrics
* PSNR callback
* SSIM callback

## Running and Stop
* ModelCheckpoint
* EarlyStopping

## Possible Modifications
* Use of RGB instead of YUV
* Loss functions:
    * MSE
    * Pixel-wise
    * Contet Loss
    * Total Variation Loss
    * Adversarial Loss
* Layers:
    * Pooling
    * Skips
    * Dense blocks
* Model structure:
    * Pre-upsampling
    * post-upsampling
    * progressiv-upsampling
    * Iterative upscaling and downscaling
    * Auto Encoder
* [May be worth a look](https://beyondminds.ai/blog/an-introduction-to-super-resolution-using-deep-learning/)


