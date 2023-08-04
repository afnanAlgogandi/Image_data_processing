import cv2
import glob
import os


# Load the cascade
dog_cascade = cv2.CascadeClassifier('haarcascade_models\haarcascade_frontalface_default.xml')

# Define the output directory
output_dir = "cropedData"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop over all dog images
for img_path in glob.glob("data/*.jpg"):
    
    # Read the image
    img = cv2.imread(img_path)
    
    # Convert into grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    dogs = dog_cascade.detectMultiScale(img, 1.1, 4)
    
    # If no dogs are detected, continue to the next image
    if len(dogs) == 0:
        continue
    
    # Assuming there's only one dog in the image, get its face
    x, y, w, h = dogs[0]
    
    # Crop the image to the dog's face
    crop_img = img[y:y+h, x:x+w]
    
    # Create the output path
    base_name = os.path.basename(img_path)
    output_path = os.path.join(output_dir, base_name)
    
    # Save the cropped image to the output directory
    cv2.imwrite(output_path, crop_img)