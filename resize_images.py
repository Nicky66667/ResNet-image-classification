from PIL import Image
import os

input_folder = 'data_set/rgb_frame/train/v_Archery_g24_c01'
output_folder = 'data_set/rgb_frame/train/Archery'
new_size = (256, 256)

# Loop through all the files in the input folder
for filename in os.listdir(input_folder):
    # Check if the file is an image
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # Open the image and resize it
        img = Image.open(os.path.join(input_folder, filename))
        img = img.resize(new_size)

        # Save the resized image to the output folder
        new_filename = os.path.splitext(filename)[0] + '.jpg'
        img.save(os.path.join(output_folder, new_filename))