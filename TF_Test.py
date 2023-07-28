import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

# Load the model
model = tf.keras.models.load_model('model.h5``')

# Load the test images that are any image in the Test folder png/img
test_images = []
for filename in glob.glob('TestData/*.png'):
    test_images.append(cv2.imread(filename))
for filename in glob.glob('TestData/*.jpg'):
    test_images.append(cv2.imread(filename))
for filename in glob.glob('TestData/*.jpeg'):
    test_images.append(cv2.imread(filename))


print(len(test_images))

# Loop through the test images and resize them to 224x224
for i in range(len(test_images)):
    test_images[i] = cv2.resize(test_images[i], (224, 224))

# Convert the images into numpy arrays so that we can feed them into the model
test_images = np.array(test_images)

print(test_images.shape)

# Predict the test images
predictions = model.predict(test_images)

print(predictions)

# Use the argmax function to get the index of the highest probability for each image
predictions = np.argmax(predictions, axis=1)

print(predictions)

# If the index is 0, then the rotation is 0 degrees, if the index is 1, then the rotation is 90 degrees
# If the index is 2, then the rotation is 180 degrees, if the index is 3, then the rotation is 270 degrees
# Use this information to print the rotation of each image and display the images side by side
for i in range(len(predictions)):
    rotation = 0
    if predictions[i] == 0:
        print("Image " + str(i) + " is rotated 0 degrees")
    elif predictions[i] == 1:
        print("Image " + str(i) + " is rotated 180 degrees")
        rotation = 180
    elif predictions[i] == 2:
        print("Image " + str(i) + " is rotated 270 degrees")
        rotation = 270
    elif predictions[i] == 3:
        print("Image " + str(i) + " is rotated 90 degrees")
        rotation = 90

    # Plot the original image and the rotated image side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(test_images[i], cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    if rotation == 0:
        rotated_image = test_images[i]
    elif rotation == 90:
        rotated_image = cv2.rotate(test_images[i], cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        rotated_image = cv2.rotate(test_images[i], cv2.ROTATE_180)
    elif rotation == 270:
        rotated_image = cv2.rotate(test_images[i], cv2.ROTATE_90_COUNTERCLOCKWISE)

    axes[1].imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Was Rotated " + str(rotation) + " Degrees")
    axes[1].axis('off')

    plt.show()
