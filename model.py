"""
Model architecture definition for Resistor Band Detection and Classification.
Provides MobileNetV3/MobileNetV2 feature backbone with Faster R-CNN detection heads.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

NUM_CLASSES = 10  # 9 band classes + 1 background class

def get_resistor_detector(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """
    Creates a Faster R-CNN object detector with a MobileNet backbone
    optimized for fast mobile/CPU inference and accurate color band localization.
    """
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights,
        min_size=256,
        max_size=320,
        box_score_thresh=0.05,
        box_nms_thresh=0.3,
        box_detections_per_img=10
    )
    
    # Replace the box predictor head with the custom number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
