import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet34  # Import the ResNet34 model from the model file


def main():
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # Define the data transformation pipeline for training and validation
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # Set the root directory for the dataset
    data_root = os.path.abspath(os.path.join(os.getcwd(), "/home/nicky/git/Resnet image classification/"))  # get data root path
    # Path to the flower dataset
    image_path = os.path.join(data_root, "data_set/", "flower_data")
    # Ensure the dataset path exists
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    # Load training dataset and apply the defined transformations
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    # Get the number of training images
    train_num = len(train_dataset)

    # Mapping from class labels to class names
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # Set batch size and number of workers for data loading
    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # DataLoader for training and validation datasets
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    # Load validation dataset and apply the defined transformations
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    # Print the number of images used for training and validation
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # Initialize the ResNet34 model
    net = resnet34()


    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet34-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    # Replace the fc layer to output 5 classes (flower types)
    net.fc = nn.Linear(in_channel, 5)
    # Move the model to the appropriate device (GPU/CPU)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    # Set the number of training epochs
    epochs = 3
    best_acc = 0.0
    save_path = './resNet34.pth'
    train_steps = len(train_loader)  # Number of steps per epoch (for training)

    # Training loop
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        # Create a progress bar for training
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad() # Clear the gradients
            logits = net(images.to(device)) # Forward pass
            loss = loss_function(logits, labels.to(device)) # Compute the loss
            loss.backward()
            optimizer.step() # Update model weights

            # Accumulate loss for logging
            running_loss += loss.item()

            # Update progress bar
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # Validation phase

        # Set the model to evaluation mode
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)  # Progress bar for validation
            for val_data in val_bar:
                val_images, val_labels = val_data # Get the batch of validation images and labels
                outputs = net(val_images.to(device)) # Forward pass on validation data
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1] # Get the predicted class (index with max output)
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item() # Accumulate correct predictions

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,epochs) # Update progress bar

        # Compute validation accuracy
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        # Save the model if the validation accuracy improves
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path) # Save the model with the best accuracy

    print('Finished Training')


if __name__ == '__main__':
    main()
