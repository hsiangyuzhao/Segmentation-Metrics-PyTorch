# Segmentation Metrics PyTorch Implementation
**Author: Hsiangyu Zhao**

## Introduction
This file provides 2 Python classes for semantic segmentation metrics calculation, including multiclass cases and binary cases. Supported metrics including pixel accuracy, Dice coeff, precision and recall (Specificity is also supported in binary cases as it is meaningless in multiclass cases). The function of providing average metric (e.g. average Dice, average precision, etc.) and ignoreing background is also supported.  
Calculation can be performed during training on batches, so tranformation from Torch Tensor to numpy array or PIL images is not needed. Normally the metrics are calculated on batches, which is fast and easy to implement but accuracy is compromised. When set batch size to one, calculation will be performed per image. We recommend setting batch size to one during inference as it provides accurate results on every image.

## Requirements
PyTorch 1.2.0  
numpy 1.17.4  
**other versions of PyTorch and numpy are not tested, so I cannot guarantee whether the codes can be run on them.**

## Features
- Provide implematation of commonly used metrics.
- Provide the argment of ignoring the background (pixels that are labeled as 0 in a mask).
- Provide 4 kind of output activation to calculate metrics, depending on users' purposes.

## Supproted Metrics
### Pixel accuracy
Pixel accuracy measures how many pixels are predicted correctly. In binary cases:  
$Pixel Acc = \frac{TP + TN}{TP + TN + FP + FN}$  
In multiclass cases it can be calculated from confusion matrix, by dividing the sum of diagonal elements (ture positives for all classes) with the total number of pixels.
### Dice coeff
Dice evaluates the overlap rate of prediction results and ground truth; equals to f1 score in defination.  
$Dice = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$  
### Precision
describes the purity of our positive detections relative to the ground truth.  
$Precision = \frac{TP}{TP + FP}$  
### Recall
describes the completeness of our positive predictions relative to the ground truth.  
$Recall = \frac{TP}{TP + FN}$  
### Specificity
Also known as true negative rate (TNR)  
$Specificity = \frac{TN}{TN + FP}$  

🛑 **Please feel free to use it and report any potential issues or bugs! Enjoy!**
