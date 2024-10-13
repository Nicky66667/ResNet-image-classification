import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    """
    Creates a directory at the given path. If the directory already exists,
    it deletes the existing directory and creates a new one.
    """
    if os.path.exists(file_path):
        # If the folder exists, remove the original folder before recreating it
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    """
    The dataset is split by a specified rate (10% validation, 90% training).
    It creates two new directories 'train' and 'val' to store the respective data.
    """

    # Ensure reproducibility of random operations
    random.seed(0)

    # Set the split rate for the validation set (10%)
    split_rate = 0.1

    # Define the paths to the flower_photos dataset and check if it exists
    cwd = os.getcwd()
    data_root = os.path.join(cwd, "flower_data")
    origin_flower_path = os.path.join(data_root, "flower_photos") #ssumes that the dataset is stored in 'flower_photos'.
    assert os.path.exists(origin_flower_path), "path '{}' does not exist.".format(origin_flower_path)

    # List all flower classes
    flower_class = [cla for cla in os.listdir(origin_flower_path)
                    if os.path.isdir(os.path.join(origin_flower_path, cla))]

    # Create the directories for training and validation sets
    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for cla in flower_class:
        # Create a folder for each flower class
        mk_file(os.path.join(train_root, cla))

    val_root = os.path.join(data_root, "val")
    mk_file(val_root) # Create or recreate the val folder

    for cla in flower_class:
        # Create a folder for each flower class
        mk_file(os.path.join(val_root, cla))

    # Process each flower class and split the data into train and validation sets
    for cla in flower_class:
        cla_path = os.path.join(origin_flower_path, cla)
        images = os.listdir(cla_path) # List all images in the class folder
        num = len(images)

        # Randomly select a portion (10%) of the images for the validation set
        eval_index = random.sample(images, k=int(num*split_rate))
        for index, image in enumerate(images):
            if image in eval_index:
                # If the image is in the validation set, copy it to the validation folder
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                # Otherwise, copy it to the training folder
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            # Print a progress bar
            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    main()
