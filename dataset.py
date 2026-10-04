"""
Dataset and Data Loading utilities for Resistor Band Detection.
Handles YOLO polygon format, converts to bounding boxes, and provides augmentations.
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Any, Optional

CLASS_NAMES = [
    'Black',        # 0 -> 1 in detector (0 = background)
    'Blue',         # 1 -> 2
    'Brown',        # 2 -> 3
    'Green',        # 3 -> 4
    'Orange',       # 4 -> 5
    'Red',          # 5 -> 6
    'Side-Gold',    # 6 -> 7
    'Side-Silver',  # 7 -> 8
    'Yellow'        # 8 -> 9
]

class ResistorBandDataset(Dataset):
    def __init__(self, data_dir: str = 'resistor_images', split: str = 'train', augment: bool = False):
        self.split = split
        self.augment = augment
        self.base_dir = os.path.join(data_dir, split)
        self.img_paths = sorted(glob.glob(os.path.join(self.base_dir, 'images', '*')))
        
    def __len__(self) -> int:
        return len(self.img_paths)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        boxes = []
        labels = []
        
        fname = os.path.splitext(os.path.basename(img_path))[0]
        lbl_file = os.path.join(self.base_dir, 'labels', fname + '.txt')
        
        if os.path.exists(lbl_file):
            with open(lbl_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 9:
                        cls_id = int(parts[0]) + 1  # 1-indexed for PyTorch Faster-RCNN
                        xs = [float(parts[1])*w, float(parts[3])*w, float(parts[5])*w, float(parts[7])*w]
                        ys = [float(parts[2])*h, float(parts[4])*h, float(parts[6])*h, float(parts[8])*h]
                        xmin = max(0.0, min(xs))
                        xmax = min(float(w), max(xs))
                        ymin = max(0.0, min(ys))
                        ymax = min(float(h), max(ys))
                        
                        if xmax > xmin + 2 and ymax > ymin + 2:
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(cls_id)
                            
        # Augmentation (simple random horizontal flip & slight color jitter for training)
        if self.augment and np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            new_boxes = []
            for b in boxes:
                new_boxes.append([w - b[2], b[1], w - b[0], b[3]])
            boxes = new_boxes
            
        if len(boxes) == 0:
            boxes = [[0.0, 0.0, 1.0, 1.0]]
            labels = [0]
            
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'area': torch.tensor([(b[2]-b[0])*(b[3]-b[1]) for b in boxes], dtype=torch.float32),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

def get_dataloader(data_dir: str = 'resistor_images', split: str = 'train', batch_size: int = 4, shuffle: bool = True):
    dataset = ResistorBandDataset(data_dir=data_dir, split=split, augment=(split == 'train'))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=0)
