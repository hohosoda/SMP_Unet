## Supervised Learning for Medical Image Segmentation

## Requirements
Some important required packages include:
* Python == 3.8
* pytorch==2.1.0
* torchvision==0.16.0
* torchaudio==2.1.0
* Basic python packages such as Numpy, OpenCV ......

# Usage
1. Data set
Dataset is arranged in the following format:
```
DATA/
|-- Ki67
|   |-- TrainDataset
|   |   |-- patch
|   |   |-- label
|   |-- InferenceDataset
|   |   |-- patch
|   |   |-- perdict
```
3. Model Structure
```
![image] (https://github.com/r08543063/SMP_Unet/blob/main/model.PNG)
```
5. Train the model
```
cd code
python SMP_Unet_train.py
```
5. Inference
```
cd code
python SMP_Unet_inference.py
```
