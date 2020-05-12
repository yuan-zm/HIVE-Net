# HIVE-Net-Centerline-Aware-HIerarchical-View-Ensemble-Convolutional-Network-for-Mitochondria-Segment
Here are implementations for paper: <br />

**HIVE-Net: Centerline-Aware HIerarchical View-Ensemble Convolutional Network for Mitochondria Segmentation in EM Images**.

Contact: Zhimin Yuan (zhimin_yuan@163.com)

## Network Structure

![](figures/network_architucture.pdf) 

## Requirements
- Tested with Python 3.6
- CUDA 9.0 or higher
- PyTorch 1.1.0 
- numpy 1.16.4
- albumentations 0.3.0

## Task
Mitochondria segmentation 

## Dataset
1. EPFL dataset: https://www.epfl.ch/labs/cvlab/data/data-em/
2. Kasthuri++ dataset: https://sites.google.com/view/connectomics/home

The data augmentation library [Albumentation](https://github.com/albumentations-team/albumentations)

## Training
Run `main.py` either in your favorite Python IDE or the terminal by typing:
```
python main.py
```

## Testing
Run `mito_prediction.py` either in your favorite Python IDE or the terminal by typing:
```
python mito_prediction.py
```


