from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from PIL import Image
import numpy as np

# Define the data augmentation parameters
datagen = ImageDataGenerator(
        rotation_range=30,
        shear_range=0.2,
        zoom_range=[0.8, 1.2],
        brightness_range=[0.5, 1.5],
        channel_shift_range=20.0,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

# Define the path to the directory containing the images you want to augment
input_dir = 'cropedData'

# Define the path to the directory where you want to save the augmented images
output_dir = 'data'

# Loop over each image file in the input directory and augment it
for filename in os.listdir(input_dir):
    # Load the image and convert it to grayscale
    img = load_img(os.path.join(input_dir, filename), color_mode='grayscale')

    # Convert the image to a numpy array
    x = img_to_array(img)

    # Reshape the array to a 4D tensor with shape (1, height, width, channels)
    x = x.reshape((1,) + x.shape)

    # Generate batches of augmented images
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix=filename[:-4] + '_aug', save_format='jpg'):
        i += 1
        if i > 9:  # Generate 10 augmented images per original image
            break
        