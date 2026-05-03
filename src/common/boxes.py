import torch
from torchvision.ops import box_iou


def generalized_box_iou(boxes1, boxes2):
    iou = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))[:, None]
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))[None]
    lt_i = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb_i = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_i = (rb_i - lt_i).clamp(min=0)
    inter = wh_i[:, :, 0] * wh_i[:, :, 1]
    union = area1 + area2 - inter
    return iou - (area - union) / area.clamp(min=1e-6)

