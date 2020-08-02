import numpy as np
import torch
import torch.nn as nn


class SegmentationMetrics(object):
    r"""Calculate common metrics in semantic segmentation to evalueate model preformance.

    Supported metrics: Dice Coeff, IoU (Intersection over Union),
    precision score and recall score.

    Dice Coeff is a measure function to measure similarity over 2 sets, which is usually used to
    calculate the similarity of two samples. Dice equals to f1 score in semantic segmentation tasks.

    Intersection over Union, also referred to as the Jaccard index, is essentially a method to
    quantify the percent overlap between the target mask and our prediction output. This metric
    is closely related to the Dice coefficient.

    Precision describes the purity of our positive detections relative to the ground truth. Of all
    the objects that we predicted in a given image, precision score describes how many of those objects
    actually had a matching ground truth annotation.

    Recall describes the completeness of our positive predictions relative to the ground truth. Of
    all the objected annotated in our ground truth, recall score describes how many true positive instances
    we have captured in semantic segmentation.

    Args:
        eps: float, a value added to the denominator for numerical stability.
            Default: 1e-5

        average: string, [None, 'binary', 'macro' (default), 'micro', 'weighted']
            This parameter is required for multiclass/multilabel targets.
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:

            ``'binary'``:
            
                Only report results for positive class, in this case class 1.
                This is applicable only if `y_true` and `y_pred` are binary,
                where 0 denotes negative class and 1 denotes positive class.
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score (dice coeff) that is not between precision and recall.

        ignore_background: bool. Default: ``True``
            When set to ``True``, the class will not calculate related metrics on
            background pixels. When the segmentation of background pixels is not
            important, set this value to ``True``.
            This argment is ignored if set `average='binary'`.

        activation: [None, 'none', 'softmax' (default), 'sigmoid', '0-1']
            This parameter determines what kind of activation function that will be
            applied on model output.

    Input:
        y_true: :math:`(N, H, W)`, torch tensor, where we use int value between (0, num_class - 1)
        to denote every class, where ``0`` denotes background class.
        y_pred: :math:`(N, C, H, W)`, torch tensor.

    Examples::
        >>> mask = [[[0, 1], [0, 2], [2, 1], [1, 1]], [[1, 2], [2, 0], [1, 1], [2, 2]]]
        >>> y_true = torch.as_tensor(mask)
        >>> y_pred = torch.randn(2, 3, 4, 2)
        >>> metric_calculator = SegmentationMetrics(average='macro', ignore_background=True)
        >>> dice, iou, precision, recall = metric_calculator(y_true, y_pred)
    """
    def __init__(self, eps=1e-5, average='macro', ignore_background=True, activation='softmax'):
        assert average in [None, 'none', 'binary', 'micro', 'macro', 'weighted']

        self.eps = eps
        self.average = average
        self.ignore = ignore_background
        self.activation = activation

    @staticmethod
    def _one_hot(gt, pred, class_num):
        # transform sparse mask into one-hot mask
        # shape: (B, H, W) -> (B, C, H, W)
        one_hot = torch.zeros((gt.shape[0], class_num, gt.shape[1], gt.shape[2])).to(pred.device, dtype=torch.float)
        target = one_hot.scatter_(1, gt.unsqueeze(1).long().data, 1.0)
        return target

    @staticmethod
    def _get_class_data(gt_onehot, pred, class_num):
        # perform calculation on a batch
        # for precise result in a single image, plz set batch size to 1
        matrix = np.zeros((3, class_num))
        weights = np.zeros(class_num)

        # calculate tp, fp, fn per class
        for i in range(class_num):
            # pred shape: (N, H, W)
            class_pred = pred[:, i, :, :]
            # gt shape: (N, H, W), binary array where 0 denotes negative and 1 denotes positive
            class_gt = gt_onehot[:, i, :, :]

            pred_flat = class_pred.contiguous().view(-1, )  # shape: (N * H * W, )
            gt_flat = class_gt.contiguous().view(-1, )  # shape: (N * H * W, )

            weight = torch.sum(gt_flat)

            tp = torch.sum(gt_flat * pred_flat)
            fp = torch.sum(pred_flat) - tp
            fn = torch.sum(gt_flat) - tp

            matrix[:, i] = tp.item(), fp.item(), fn.item()
            weights[i] = weight

        return matrix, weights

    def _calculate_binary_metrics(self, gt, pred, class_num):
        # calculate metrics in binary-class segmentation
        matrix, _ = self._get_class_data(gt, pred, class_num)
        matrix = matrix[:, 1:]

        tp = np.sum(matrix[0, :])
        fp = np.sum(matrix[1, :])
        fn = np.sum(matrix[2, :])
        dice = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
        iou = (tp + self.eps) / (tp + fp + fn + self.eps)
        precision = (tp + self.eps) / (tp + fp + self.eps)
        recall = (tp + self.eps) / (tp + fn + self.eps)

        return dice, iou, precision, recall

    def _calculate_multi_metrics(self, gt, pred, class_num):
        # calculate metrics in multi-class segmentation
        matrix, weights = self._get_class_data(gt, pred, class_num)
        if self.ignore:
            matrix = matrix[:, 1:]
            weights = weights[1:]

        if self.average == 'micro':
            tp = np.sum(matrix[0, :])
            fp = np.sum(matrix[1, :])
            fn = np.sum(matrix[2, :])
            dice = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
            iou = (tp + self.eps) / (tp + fp + fn + self.eps)
            precision = (tp + self.eps) / (tp + fp + self.eps)
            recall = (tp + self.eps) / (tp + fn + self.eps)

        elif self.average in [None, 'none']:
            dice = (2 * matrix[0, :] + self.eps) / (2 * matrix[0, :] + matrix[1, :] + matrix[2, :] + self.eps)
            iou = (matrix[0, :] + self.eps) / (matrix[0, :] + matrix[1, :] + matrix[2, :] + self.eps)
            precision = (matrix[0, :] + self.eps) / (matrix[0, :] + matrix[1, :] + self.eps)
            recall = (matrix[0, :] + self.eps) / (matrix[0, :] + matrix[2, :] + self.eps)

        else:
            dice_list = (2 * matrix[0, :] + self.eps) / (2 * matrix[0, :] + matrix[1, :] + matrix[2, :] + self.eps)
            iou_list = (matrix[0, :] + self.eps) / (matrix[0, :] + matrix[1, :] + matrix[2, :] + self.eps)
            precision_list = (matrix[0, :] + self.eps) / (matrix[0, :] + matrix[1, :] + self.eps)
            recall_list = (matrix[0, :] + self.eps) / (matrix[0, :] + matrix[2, :] + self.eps)
            if self.average == 'weighted':
                dice_list = dice_list * weights / np.sum(weights)
                iou_list = iou_list * weights / np.sum(weights)
                precision_list = precision_list * weights / np.sum(weights)
                recall_list = recall_list * weights / np.sum(weights)
            dice = np.average(dice_list)
            iou = np.average(iou_list)
            precision = np.average(precision_list)
            recall = np.average(recall_list)

        return dice, iou, precision, recall

    def __call__(self, y_true, y_pred):
        class_num = y_pred.size(1)

        if self.activation in [None, 'none']:
            activation_fn = lambda x: x
            activated_pred = activation_fn(y_pred)
        elif self.activation == "sigmoid":
            activation_fn = nn.Sigmoid()
            activated_pred = activation_fn(y_pred)
        elif self.activation == "softmax":
            activation_fn = nn.Softmax(dim=1)
            activated_pred = activation_fn(y_pred)
        elif self.activation == "0-1":
            pred_argmax = torch.argmax(y_pred, dim=1)
            activated_pred = self._one_hot(pred_argmax, y_pred, class_num)
        else:
            raise NotImplementedError("Not a supported activation!")

        gt_onehot = self._one_hot(y_true, y_pred, class_num)
        if self.average == 'binary':
            assert activated_pred.shape[1] == 2  # Targets must be binary when set `average='binary'`
            dice, iou, precision, recall = self._calculate_binary_metrics(gt_onehot, activated_pred, class_num)
        else:
            dice, iou, precision, recall = self._calculate_multi_metrics(gt_onehot, activated_pred, class_num)
        return dice, iou, precision, recall
