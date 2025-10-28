# # Copyright (c) OpenMMLab. All rights reserved.
# import warnings

# import torch
# import torch.nn as nn
# from mmdet.models.losses.utils import weighted_loss

# from mmrotate.registry import MODELS

# try:
#     from mmcv.ops import diff_iou_rotated_2d
# except:  # noqa: E722
#     diff_iou_rotated_2d = None


# @weighted_loss
# def rotated_iou_loss(pred, target, linear=False, mode='log', eps=1e-6):
#     """Rotated IoU loss.

#     Computing the IoU loss between a set of predicted rbboxes and target
#      rbboxes.
#     The loss is calculated as negative log of IoU.
#     Args:
#         pred (torch.Tensor): Predicted bboxes of format (x, y, h, w, angle),
#             shape (n, 5).
#         target (torch.Tensor): Corresponding gt bboxes, shape (n, 5).
#         linear (bool, optional): If True, use linear scale of loss instead of
#             log scale. Default: False.
#         mode (str): Loss scaling mode, including "linear", "square", and "log".
#             Default: 'log'
#         eps (float): Eps to avoid log(0).
#     Return:
#         torch.Tensor: Loss tensor.
#     """
#     assert mode in ['linear', 'square', 'log']
#     if linear:
#         mode = 'linear'
#         warnings.warn(
#             'DeprecationWarning: Setting "linear=True" in '
#             'poly_iou_loss is deprecated, please use "mode=`linear`" '
#             'instead.')

#     if diff_iou_rotated_2d is None:
#         raise ImportError('Please install mmcv-full >= 1.5.0.')

#     ious = diff_iou_rotated_2d(pred.unsqueeze(0), target.unsqueeze(0))
#     ious = ious.squeeze(0).clamp(min=eps)

#     if mode == 'linear':
#         loss = 1 - ious
#     elif mode == 'square':
#         loss = 1 - ious**2
#     elif mode == 'log':
#         loss = -ious.log()
#     else:
#         raise NotImplementedError
#     return loss


# @MODELS.register_module()
# class RotatedIoULoss(nn.Module):
#     """RotatedIoULoss.

#     Computing the IoU loss between a set of predicted rbboxes and
#     target rbboxes.
#     Args:
#         linear (bool): If True, use linear scale of loss else determined
#             by mode. Default: False.
#         eps (float): Eps to avoid log(0).
#         reduction (str): Options are "none", "mean" and "sum".
#         loss_weight (float): Weight of loss.
#         mode (str): Loss scaling mode, including "linear", "square", and "log".
#             Default: 'log'
#     """

#     def __init__(self,
#                  linear=False,
#                  eps=1e-6,
#                  reduction='mean',
#                  loss_weight=1.0,
#                  mode='log'):
#         super(RotatedIoULoss, self).__init__()
#         assert mode in ['linear', 'square', 'log']
#         if linear:
#             mode = 'linear'
#             warnings.warn('DeprecationWarning: Setting "linear=True" in '
#                           'IOULoss is deprecated, please use "mode=`linear`" '
#                           'instead.')
#         self.mode = mode
#         self.linear = linear
#         self.eps = eps
#         self.reduction = reduction
#         self.loss_weight = loss_weight

#     def forward(self,
#                 pred,
#                 target,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None,
#                 **kwargs):
#         """Forward function.

#         Args:
#             pred (torch.Tensor): The prediction.
#             target (torch.Tensor): The learning target of the prediction.
#             weight (torch.Tensor, optional): The weight of loss for each
#                 prediction. Defaults to None.
#             avg_factor (int, optional): Average factor that is used to average
#                 the loss. Defaults to None.
#             reduction_override (str, optional): The reduction method used to
#                 override the original reduction method of the loss.
#                 Defaults to None. Options are "none", "mean" and "sum".
#         """
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#         if (weight is not None) and (not torch.any(weight > 0)) and (
#                 reduction != 'none'):
#             if pred.dim() == weight.dim() + 1:
#                 weight = weight.unsqueeze(1)
#             return (pred * weight).sum()  # 0
#         if weight is not None and weight.dim() > 1:
#             # TODO: remove this in the future
#             # reduce the weight of shape (n, 5) to (n,) to match the
#             # iou_loss of shape (n,)
#             assert weight.shape == pred.shape
#             weight = weight.mean(-1)
#         loss = self.loss_weight * rotated_iou_loss(
#             pred,
#             target,
#             weight,
#             mode=self.mode,
#             eps=self.eps,
#             reduction=reduction,
#             avg_factor=avg_factor,
#             **kwargs)
#         return loss




import torch
import torch.nn as nn
import warnings
from mmdet.models.losses.utils import weighted_loss
from mmrotate.registry import MODELS

# 引入 mmcv 的 diff_iou_rotated_2d 函數
try:
    from mmcv.ops import diff_iou_rotated_2d
except ImportError:
    diff_iou_rotated_2d = None

@weighted_loss
def rotated_iou_loss_modified(pred, target, weight=None, mode='log', eps=1e-6):
    """
    修改後的 rotated_iou_loss:
      - 原始的 ious 保持不變 (使用 diff_iou_rotated_2d)。
      - 額外計算：
          D_boundary_norm = D_boundary / L_diagonal，
            其中 D_boundary 為 pred 與 target 各四個頂點間的平均歐式距離，
            L_diagonal 為兩組框所有頂點組成的最小外接矩形對角線長度。
          D_center_norm = center_distance / L_diagonal，
            其中 center_distance 為 pred 與 target 的中心點距離。
      - 定義增強後的 IoU 為:
          augmented_iou = ious - (lambda1 * D_boundary_norm + lambda2 * D_center_norm)
      - 根據 mode 計算損失：
          linear: loss = 1 - augmented_iou
          square: loss = 1 - (augmented_iou)^2
          log:    loss = -log(augmented_iou)
    """
    if diff_iou_rotated_2d is None:
        raise ImportError('Please install mmcv-full >= 1.5.0.')


    ious = diff_iou_rotated_2d(pred.unsqueeze(0), target.unsqueeze(0))
    ious = ious.squeeze(0).clamp(min=eps)  # shape: (N,)


    def get_rotated_box_corners(box):
        # box: (N, 5)
        x, y, w, h, angle = box.unbind(dim=1)
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        # 定義偏移量，順序：左上、右上、右下、左下
        dx = torch.stack([-w/2, w/2, w/2, -w/2], dim=1)
        dy = torch.stack([-h/2, -h/2, h/2, h/2], dim=1)
        # 旋轉偏移量
        rotated_dx = cos.unsqueeze(1) * dx - sin.unsqueeze(1) * dy
        rotated_dy = sin.unsqueeze(1) * dx + cos.unsqueeze(1) * dy
        corners = torch.stack([x.unsqueeze(1) + rotated_dx, y.unsqueeze(1) + rotated_dy], dim=2)
        return corners  # shape: (N, 4, 2)


    pred_corners = get_rotated_box_corners(pred)
    target_corners = get_rotated_box_corners(target)


    boundary_diff = torch.norm(pred_corners - target_corners, dim=2)  # (N, 4)
    D_boundary = boundary_diff.mean(dim=1)  # (N,)

    all_corners = torch.cat([pred_corners, target_corners], dim=1)  # (N, 8, 2)
    min_xy = all_corners.min(dim=1)[0]  # (N, 2)
    max_xy = all_corners.max(dim=1)[0]  # (N, 2)
    
    # L_diagonal = torch.sqrt(((max_xy - min_xy) ** 2).sum(dim=1))  #對角線
    L_diagonal = ((max_xy - min_xy) ** 2).sum(dim=1)  #對角線平方    

    # 標準化邊界距離
    D_boundary_norm = D_boundary / (L_diagonal + eps)

    # 計算中心點距離並標準化
    center_pred = pred[:, :2]  # (N, 2)
    center_target = target[:, :2]  # (N, 2)
    D_center = torch.norm(center_pred - center_target, dim=1)  # (N,)
    D_center_norm = D_center / (L_diagonal + eps)

    # 定義增強後的 IoU 值
    # augmented_iou = ious - (  D_boundary_norm + D_center_norm )

    # augmented_iou = ious - ( D_boundary_norm )

    augmented_iou = ious - ( D_center_norm )

    # augmented_iou =  D_boundary_norm 

    # 根據 mode 計算損失
    if mode == 'linear':
        loss = 1 - augmented_iou
    elif mode == 'square':
        loss = 1 - augmented_iou**2
    elif mode == 'log':
        loss = -torch.log(augmented_iou)
    else:
        raise NotImplementedError(f"Mode {mode} is not supported.")
    return loss

@MODELS.register_module()
class RotatedIoULoss(nn.Module):
    def __init__(self,
                 mode='linear',
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0):
        super(RotatedIoULoss, self).__init__()
        self.mode = mode
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if (weight is not None) and (not torch.any(weight > 0)) and (reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # 將 weight 形狀 (n, 5) 轉成 (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * rotated_iou_loss_modified(
            pred,
            target,
            weight=weight,
            mode=self.mode,
            eps=self.eps,
            avg_factor=avg_factor,
            **kwargs)
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
