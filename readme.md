# Segmentation Metrics PyTorch Implementation
**Author: Hsiangyu Zhao**

## Introduction
This file provides a Python class for semantic segmentation metrics calculation. Supported metrics including Dice coeff, intersection over union (IoU), precision and recall. All metrics supprot 4 kinds of average: binary, micro, macro and weighted.  
Calculation can be performed during training on batches, so tranformation from Torch Tensor to numpy array or PIL images is not needed. When set batch size to one, calculation will be performed per image.  

## Requirements
PyTorch 1.2.0  
numpy 1.17.4  
**other versions of PyTorch and numpy are not tested, so I cannot guarantee whether the codes can be run on them.**

## Features
- Provide implematation of 4 commonly used metrics.
- Provide 4 kinds of average, and users can choose to use any kind of average depending on their purposes.
- Provide the argment of ignoring the background (pixels that are labeled as 0 in a mask).
- Provide 4 kind of output activation to calculate metrics, depending on users' purposes.

## Supproted Metrics
### Dice coeff
Dice evaluates the overlap rate of prediction results and ground truth; equals to f1 score in defination.  
$Dice = \frac{2 \cdot TP}{2 \cdot TP + FP + Fn}$  
### IoU
Also known as Jaccard index, measures similarity between finite sample sets, and is defined as the size of the intersection divided by the size of the union of the sample sets. (Wikipedia)  
$IoU = \frac{TP}{TP + FP + FN}$  
### Precision
describes the purity of our positive detections relative to the ground truth.  
$Precision = \frac{TP}{TP + FP}$  
### Recall
describes the completeness of our positive predictions relative to the ground truth.  
$Recall = \frac{TP}{TP + FN}$  

## Supported Average
including binary, micro, macro and weighted average.
### binary
Only report results for positive class, in this case class 1.
### micro
Calculate metrics globally by counting the total true positives, false negatives and false positives.
### macro
Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
### weighted
Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).  
This alters 'macro' to account for label imbalance; it can result in an F-score (dice coeff) that is not between precision and recall.

🛑 **Please feel free to use it and report any potential issues or bugs! Enjoy!**
