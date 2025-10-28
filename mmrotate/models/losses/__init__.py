# Copyright (c) OpenMMLab. All rights reserved.
from .rotated_iou_loss import RotatedIoULoss
from .smooth_l1_loss import MYSmoothL1Loss
from .cross_entropy_loss import MYCrossEntropyLoss


__all__ = ['RotatedIoULoss','MYSmoothL1Loss','MYCrossEntropyLoss']
