# Image Rotation Classification using VGG16

This code implements a deep learning model for classifying rotated images into four categories: 0 degrees, 90 degrees, 180 degrees, and 270 degrees. The model architecture is based on the VGG16 convolutional neural network, which has been pre-trained on the ImageNet dataset. The code fine-tunes the VGG16 model for the specific image rotation classification task using transfer learning.

## Requirements

To run this code, you need the following dependencies:

- TensorFlow
- Keras
- tqdm
- NumPy
- cv2
- matplotlib.pyplot
- glob

Please ensure that you have installed these libraries before running the code.

## Dataset

Before running the code, you need to prepare your dataset in a specific directory structure. Running genRotatedDataset.py and filling in the variables for ```dataset_path ``` and ```output_dir ``` enable you to control the input and output directories 

### Output Directory Structure:

```
Dataset/
    |-- Rotated Images/
            |-- 0_Degrees/
                |-- 0_img1.jpg
                |-- 0_img2.jpg
                |-- ...
            |-- 90_Degrees/
                |-- 90_img1.jpg
                |-- 90_img2.jpg
                |-- ...
            |-- 180_Degrees/
                |-- 180_img1.jpg
                |-- 180_img2.jpg
                |-- ...
            |-- 270_Degrees/
                |-- 270_img1.jpg
                |-- 270_img2.jpg
                |-- ...
```

## How to Train

1. Prepare your dataset in the specified directory structure (See above).

2. Adjust the `data_dir`,`save_path`, `batch_size`, and `num_epochs` variables as needed in the file TF_Train.py.

3. Run the code to train the model. The training progress will be displayed in the terminal.

4. After training, the model will be saved to the directory specified by the `save_path`.

**GPU Note:**

Training a deep learning model can be computationally intensive, especially with large datasets and complex architectures. For faster training, it is recommended to run this code on a machine with a GPU. The code checks for the availability of GPUs and prints the number of available GPUs at the beginning of execution. If no GPU is available, the training will still proceed but may take longer.


## How to Use for Automatic Rotation Correctiion

1. Adjust the following variables in the code as needed:

   - `model.h5`: Replace with the path to your trained model file (`model.h5` is a placeholder).
   - `TestData/*.png`, `TestData/*.jpg`, `TestData/*.jpeg`: Modify the file paths to match your test image directory.

2. Run the code to predict the rotation of the test images. The predicted rotation angles will be displayed in the terminal.

3. For each test image, the code will show the original image and the image after applying the predicted rotation side by side.

## Note about Classes:

The classes are a bit confusing.

The class it outputs says the following:

| Index      | Photo's Rotation | Corrective Rotation      |
| :---        |    :----:   |          ---: |
| 0      | 0°       | 0°  |
| 1   | 180°        | 180°       |
| 2      | 270°       | 90° Counter Clockwise   |
| 3      | 90°       | 90° Clockwise  |

So if the image is rotated 90°, the max index in the prediction will be 3 and the direction we'd need to rotate is 90° Clockwise. 

The Model Predicts the angle that the image is rotated NOT THE ANGLE NEEDED TO CORRECT THE ROTATION.


# Pretrained Model

Please use the model ```modelInsta.h5``` for your use cases and note the above class output for your use case.

## Dataset
The dataset used for training and validation was a 300,000 image subset of the Instagram dataset, agumented as mentioned in the above Dataset section. [Kaggle Link](https://www.kaggle.com/datasets/shmalex/instagram-images) The Dataset was split 80% for training, 20% validation. 

## Accuracy
| DataSet      |Accuracy | Loss|
| :---        |    :----:   |:----:  |
| Flickr30K | 0.93521| 1.15195
| COCO2017 Test Data |  0.89958       |2.46999
| Personal Family Photos (2,000)| 0.85767| 1.07506

Based on testing with my own family photos the accuracy is high for general personal photos.

## Model Architecture

The model architecture is based on the VGG16 pre-trained model. The VGG16 model has been loaded with weights from 'imagenet' and the top (classification) layers have been excluded. A custom top layer has been added to perform the final classification.

The model consists of the following layers:

1. Input Layer: 224x224x3 (3-channel RGB images)
2. Base VGG16 Model: Pre-trained VGG16 model with 'imagenet' weights, excluding the top (classification) layers.
3. Flatten Layer: Flattens the output of the VGG16 model.
4. Fully Connected Layer: Dense layer with 4096 neurons and ReLU activation.
5. Batch Normalization Layer: Applied after the first fully connected layer.
6. Dropout Layer: Dropout with a rate of 0.7 after the first fully connected layer.
7. Fully Connected Layer: Dense layer with 4096 neurons and ReLU activation.
8. Batch Normalization Layer: Applied after the second fully connected layer.
9. Dropout Layer: Dropout with a rate of 0.7 after the second fully connected layer.
10. Output Layer: Dense layer with 'num_classes' neurons and Softmax activation for classification.
