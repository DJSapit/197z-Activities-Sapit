import torch
from pytorch_lightning import Callback
from torchvision.ops import box_iou
from config import class_labels


def collate_fn(batch):
    return tuple(zip(*batch))


def evaluate_iou(target, pred, cft):
    device = pred["boxes"].device
    if pred["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return torch.tensor(0.0, device=device)

    labels2 = pred["labels"]
    scores = pred["scores"]
    labels2 = labels2[(scores >= cft)]

    if labels2.shape[0] == 0:
        # no box detected with score > cft, 0 IOU
        return torch.tensor(0.0, device=device)
    
    boxes1 = target["boxes"]
    labels1 = target["labels"]

    boxes2 = pred["boxes"]
    boxes2 = boxes2[(scores >= cft)]

    labelset1 = set(labels1.tolist())
    labelset2 = set(labels2.tolist())
    labels_same = labelset1 & labelset2

    classified_boxes1 = [boxes1[labels1 == i] for i in labels_same]
    classified_boxes2 = [boxes2[labels2 == i] for i in labels_same]

    divisor = len(labelset1 | labelset2)
    # mean IOU of detections with unmatched labels with ground truth
    return torch.tensor([box_iou(box1, box2).mean() for box1, box2 in zip(classified_boxes1,classified_boxes2)], device=device).sum() / divisor


def pred2boxes(predictions, ct=0):
    # List[Dict[Tensor]] -> Dict: boxes (FloatTensor[N, 4]), labels (Int64Tensor[N]), scores (Tensor[N])
    boxes = predictions["boxes"]
    labels = predictions["labels"]
    scores = predictions["scores"]
    num_detections = labels.shape[0]
    boxes = [
              {
              "position":{
                          "minX":boxes[i,0].item(),
                          "maxX":boxes[i,2].item(),
                          "minY":boxes[i,1].item(),
                          "maxY":boxes[i,3].item()
                          },
              "class_id" : labels[i].item(),
              "box_caption" : f"{class_labels[labels[i].item()]} {scores[i].item():0.2}", # string
              "domain" : "pixel"
              }
              for i in range(num_detections) if (labels[i].item() != 0) and (scores[i].item() >= ct)
            ]

    return {"predictions":{"box_data": boxes,"class_labels": class_labels}}


def target2boxes(target):
    # List[Dict[Tensor]] -> Dict: boxes (FloatTensor[N, 4]), labels (Int64Tensor[N])
    boxes = target["boxes"]
    labels = target["labels"]
    num_detections = labels.shape[0]
    boxes = [
              {
              "position":{
                          "minX":boxes[i,0].item(),
                          "maxX":boxes[i,2].item(),
                          "minY":boxes[i,1].item(),
                          "maxY":boxes[i,3].item()
                        },
              "class_id" : labels[i].item(),
              "box_caption" : f"{class_labels[labels[i].item()]}", # string
              "domain" : "pixel"
              }
              for i in range(num_detections) if labels[i].item() != 0
            ]

    return {"predictions":{"box_data": boxes,"class_labels": class_labels}}