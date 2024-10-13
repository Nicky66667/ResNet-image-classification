import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnet34 # Import the ResNet34 model from the model file


def main():
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the data transformation pipeline
    data_transform = transforms.Compose(
        [transforms.Resize(256), # Resize the image to 256x256
         transforms.CenterCrop(224), # Crop the center 224x224 region
         transforms.ToTensor(), # Convert the image to a tensor
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) # Normalize the image using ImageNet statistics

    # Load the image for prediction
    img_path = "./data_set/flower_data/predict_single_image.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path) # Open the image using PIL
    plt.imshow(img)  # Display the image

    # [N, C, H, W] Apply the transformations to the image
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    # Read the class indices from the JSON file
    with open(json_path, "r") as f:
        class_indict = json.load(f) # Load class indices mapping

    # Create the ResNet34 model with output classes (number of classes in your dataset)
    model = resnet34(num_classes=5).to(device)

    # load model weights
    weights_path = "./resNet34.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device)) # Load the model weights into the model

    # prediction

    # Set the model to evaluation mode (disables dropout and batch normalization updates)
    model.eval()
    with torch.no_grad():
        # Perform the prediction on the image

        # Perform inference and move to CPU for further processing
        output = torch.squeeze(model(img.to(device))).cpu()
        # Apply softmax to get class probabilities
        predict = torch.softmax(output, dim=0)
        # Get the index of the class with the highest probability
        predict_cla = torch.argmax(predict).numpy()

    # Print the predicted class and its probability
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    # Display the result on the image plot
    plt.title(print_res)

    # Print the class probabilities for all classes
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show() # Show the image with the prediction result as the title


if __name__ == '__main__':
    main()
