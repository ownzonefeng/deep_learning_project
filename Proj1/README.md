# EPFL - Deep Learning Project 1 
**`Team Members`:** Compare two digits with siamese network
**`Team Members`:** FENG Wentao, SUN Zhaodong, WANG Yunbei

In this project, we first build a convolution neural network to recognize hand-written digits. Then, based on this model, we have built a siamese convolution neural network to compare two visible digits. We also apply weight sharing and auxiliary loss to achieve the objective. Finally, we discuss the influence of those techniques.

**Instructions**:
1. Please make sure that the `PyTorch` is installed and the internet connection is established.
2. Run `test.py`, the process training will be printed on the screen and saved to `train_records.txt`



The followings are some descriptions for the modules or functions we used or designed for this project.


## Modules
### `test.py`
This is the main module. Running this module will start 20 rounds training for each combination of weight sharing status and auxiliary loss status.
