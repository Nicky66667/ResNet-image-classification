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

The available ResNet models:(download needed)
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
conda create -n image_classification python=3.7
conda activate image_classification
pip install -r requirements.txt
```

### 2. Prepare the Dataset
1. follow the instructions in the **data_set/README.md** to download the dataset.
2. split the data into training and validation sets by **data_set/split_data.py**.

### 3. Train the Model
Read the README in 
Load your dataset, split it into training and validation sets, and train the ResNet model.
```bash
python train.py
```

### 4.Make Predictions

- For a single image prediction:
```bash
python predict.py
```

- For batch image predictions:
```bash
python batch_predict.py
```

