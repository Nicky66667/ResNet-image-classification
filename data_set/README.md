### This folder is used to store the training data directory
#### The steps:
* (1) Create a new folder named "flower_data" under the data_set folder.
* (2) Download the flower classification dataset [https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)
* (3) Extract the dataset into the "flower_data" folder.
* (4) Run the "split_data.py" script to automatically split the dataset into training (train) and validation (val) sets. 

```
├── flower_data   
       ├── flower_photos (extracted dataset folder, 3,670 samples)  
       ├── train (generated training set, 3,306 samples)  
       └── val (generated validation set, 364 samples) 
```
