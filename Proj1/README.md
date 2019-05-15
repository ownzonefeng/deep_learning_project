# EPFL - Deep Learning Project 1 
**`Project name`:** Compare two digits with siamese network

**`Team members`:** FENG Wentao, SUN Zhaodong, WANG Yunbei

In this project, we first build a convolution neural network to recognize hand-written digits. Then, based on this model, we have created a siamese convolution neural network to compare two visible numbers. We also apply weight sharing and auxiliary loss to achieve the objective. Finally, we discuss the influence of those techniques.

**Instructions**:
1. Please make sure that the `PyTorch` is installed, and the internet connection is established.
2. Run `test.py`, the process of training will be printed on the screen and saved to `train_records.txt`.
3. `report.pdf` is the report of this project.

The followings are some descriptions for the modules or functions we used or designed for this project.

## Modules
### `test.py`
This is the main module. Running this module will start 20 rounds training for each combination of weight sharing status and auxiliary loss status.

### `PairDataset.py`
This is a class extending `torch.utils.data.Dataset`. It is designed to fetch data from `dlc_practical_prologue.py` and generate torch style dataset.

### `SiameseNet.py`
This module contains the structure of our siamese convolution network.

### `SiameseNetworks_train.py`
This module contains the process of training. If it works as the main function, it will conduct a single round training of 25 epochs.

### `LeNet.py`
This module is the sub-network of siamese networks, namely, single-channel convolution networks.

### `LeNet_train.py`
This module is the process of training a single channel convolution network.

### `dlc_practical_prologue.py`
The helper from the lecturer.

### `Visualisation.ipynb`
Some visualisation scripts.

### `Architecture.pdf`
The architecture of siamese networks.

### `Plot`
The output folder of `Visualisation.ipynb`. This folder contains some plots of the training process.

### `Train data records`
The output folder of `test.py`. This folder contains raw data of the training process.