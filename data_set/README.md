## This folder is used to store the training dataset directory
### The steps:
* (1) Create a new folder named "flower_data" **under the data_set folder**.
* (2) Download the flower classification dataset [https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)
* (3) Extract the dataset into the "flower_data" folder.
* (4) Run the "split_data.py" script to automatically split the dataset into training (train) and validation (val) sets. 

For any dataset in this project, you should follow this structure:

```
├── flower_data   
       ├── flower_photos (extracted dataset folder, e.g., 3,670 samples)  
       ├── train (generated training set, e.g., 3,306 samples)  
       └── val (generated validation set, e.g., 364 samples) 
```

### Optinal Dataset
1. Human Action  Recognition Dataset: (test and train folder with classes subfolders)
link: https://www.kaggle.com/datasets/shashankrapolu/human-action-recognition-dataset

2. Satellite Image Classification(data folder with classes subfolders)
link: https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification
