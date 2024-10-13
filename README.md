# Image Classification with ResNet

## Project Overview
This project demonstrates the use of ResNet architecture for image classification tasks. 
- easily adapted code to other datasets
- various ResNet models are available
- beginner-friendly with detailed comments

The project includes:
- **ResNet model implementation**: `model.py`.
- **Training script**: `train.py`
- **Single image prediction**: `predict.py` 
- **Batch image prediction**: `batch_predict.py`

The available ResNet models (download needed):
- ResNet34
- ResNet50
- ResNet101
- ResNeXt50_32x4d
- ResNeXt101_32x8d


## How to Use

### 1. Setup Environment
We recommend using **PyCharm** as your IDE and **Anaconda** to manage the environment.

Create a new conda environment and install the necessary libraries:

```bash
conda create -n resnet-image-classification python=3.8 pytorch::pytorch=1.10.0 pytorch::torchvision=0.11.1 conda-forge::tqdm conda-forge::matplotlib 
```
Or 
```bash
conda create -n image_classification python=3.8
pip install -r requirements.txt
conda install conda-forge::tqdm
conda install conda-forge::matplotlib
```

### 2. Prepare the Dataset
1. follow the instructions in the **data_set/README.md** to download the dataset.
2. split the data into training and validation sets by **data_set/split_data.py**.

**About Image Preprocessing:**
The input image should be 256x256 pixels. During data processing, the images will be resized automatically. However, if resizing fails, you can manually run **resize_images.py** to resize the images. Remember to change the input and output folder paths in the script before running it.

### 3. Train the Model
Read the README in 
Load your dataset, split it into training and validation sets, and train the ResNet model.
```bash
python train.py
```

### 4. Make Predictions

- For a single image prediction:
```bash
python predict.py
```

- For batch image predictions:
```bash
python batch_predict.py
```

### 5. Analyse the result (Optinal)

The batch_predict.py script will store the prediction results in **dataset/predict_result** in the following format:
- image path
- Predicted class
- Probability

#### Only when After running batch_predict.py

You can analyse the result records by following:

- count the number of predictions for each class
```bash
python class_count_predictions.py
```
- calculate the number of correct predictions for a specific class(Only when train the dataset with a single class)

```bash
python single_class_accuracy.py
```

