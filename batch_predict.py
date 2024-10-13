import os
import json

import torch
from PIL import Image
from torchvision import transforms

from model import resnet34


def main():
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the image transformation pipeline (resize, center crop, normalize)
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Set the root path for the test images
    imgs_root = "./data_set/flower_data/test_batch_predict"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."

    # Collect all .jpg image paths from the test directory
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]

    # Load the class indices mapping from a JSON file
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."
    json_file = open(json_path, "r")
    class_indict = json.load(json_file) # Load the class-to-index mapping

    # create model
    model = resnet34(num_classes=5).to(device)

    # load model weights
    weights_path = "./resNet34.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device)) # Load the weights into the model

    # prediction
    model.eval()

    batch_size = 8  # batches in each prediction epoch

    f = open("data_set/predict_results.txt","w")

    # Perform batch prediction
    with torch.no_grad():  # Disable gradient computation for inference (saves memory)
        for ids in range(0, len(img_path_list) // batch_size):
            # List to hold a batch of images
            img_list = []

            # Load and preprocess the images in the current batch
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path) # Open the image file
                img = data_transform(img)  # Apply the defined transformations
                img_list.append(img) # Add the processed image to the batch list

            # Stack the list of images into a single batch tensor
            batch_img = torch.stack(img_list, dim=0)

            # Perform the prediction on the batch
            output = model(batch_img.to(device)).cpu() # Run the model on the batch and move output to CPU
            predict = torch.softmax(output, dim=1) # Apply softmax
            probs, classes = torch.max(predict, dim=1)  # Get the predicted class and probability

            # Print the predictions for each image in the current batch
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                 class_indict[str(cla.numpy())],
                                                                 pro.numpy()))

                # write the records of prediction; image path, predict class, probability
                f.writelines("image: {}  class: {}  prob: {:.3}\n".format(img_path_list[ids * batch_size + idx],
                                                                 class_indict[str(cla.numpy())],
                                                                 pro.numpy()))
            f.close()

if __name__ == '__main__':
    main()
