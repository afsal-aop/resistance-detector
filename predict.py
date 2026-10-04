"""
Inference script for Resistor Color Band Prediction & Resistance Calculation.
Accepts an image, detects bands using the trained MobileNet model,
orders the color bands, and calculates the exact resistance value.
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, List, Tuple, Union

from dataset import CLASS_NAMES
from model import get_resistor_detector
from resistance_calculator import order_and_calculate_from_detections, COLOR_HEX_MAP

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# BGR color mapping for OpenCV drawing
COLOR_BGR_MAP = {
    'black': (20, 20, 20),
    'blue': (220, 100, 20),
    'brown': (30, 70, 150),
    'green': (34, 139, 34),
    'orange': (0, 140, 255),
    'red': (30, 30, 220),
    'side-gold': (50, 180, 215),
    'side-silver': (190, 190, 190),
    'yellow': (0, 215, 255)
}

class ResistorPredictor:
    def __init__(self, weights_path: str = 'weights/resistor_detector.pth', device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.class_names = CLASS_NAMES
        self.model = get_resistor_detector(num_classes=len(self.class_names) + 1, pretrained=False)
        self.weights_loaded = False
        
        if os.path.exists(weights_path):
            print(f"[INFO] Loading trained weights from {weights_path}...")
            checkpoint = torch.load(weights_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.weights_loaded = True
            print("[INFO] Weights loaded successfully!")
        else:
            print(f"[WARN] Weights file '{weights_path}' not found. Initialized with default architecture.")
            
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_input: Union[str, np.ndarray, Image.Image], score_thresh: float = 0.35) -> Dict[str, Any]:
        """
        Runs object detection on input resistor image, extracts color bands,
        and computes resistance value.
        """
        # Load and convert image
        if isinstance(image_input, str):
            orig_img = cv2.imread(image_input)
            if orig_img is None:
                raise ValueError(f"Could not load image from {image_input}")
            rgb_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            rgb_img = np.array(image_input.convert('RGB'))
            orig_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                rgb_img = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
                orig_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            elif image_input.shape[2] == 4:
                rgb_img = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
                orig_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            else:
                rgb_img = image_input
                orig_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        else:
            raise TypeError("Unsupported image input type")

        h, w, _ = rgb_img.shape
        img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(img_tensor)

        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()

        # Filter by score
        valid_indices = np.where(scores >= score_thresh)[0]
        raw_detections = []
        for idx in valid_indices:
            cls_idx = labels[idx] - 1  # 0-indexed for class_names
            if 0 <= cls_idx < len(self.class_names):
                cls_name = self.class_names[cls_idx]
                raw_detections.append({
                    'label': cls_name,
                    'box': boxes[idx].tolist(),
                    'score': float(scores[idx])
                })

        # Filter overlapping boxes (NMS per class/cross-class)
        filtered_detections = self._filter_redundant_bands(raw_detections)

        # Fallback if no detection (e.g. if model is untrained or image is tough)
        if not filtered_detections:
            return {
                'status': 'no_bands_detected',
                'raw_detections': [],
                'detections': [],
                'image_shape': (h, w),
                'annotated_image': orig_img,
                'message': 'No color bands detected with sufficient confidence.'
            }

        # Calculate resistance
        calc_result = order_and_calculate_from_detections(filtered_detections)
        annotated_img = self._draw_annotations(orig_img.copy(), calc_result)

        return {
            'status': 'success',
            'detections': filtered_detections,
            'calculation': calc_result,
            'annotated_image': annotated_img,
            'annotated_rgb': cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
            'image_shape': (h, w)
        }

    def _filter_redundant_bands(self, detections: List[Dict[str, Any]], iou_thresh: float = 0.45) -> List[Dict[str, Any]]:
        """Keep highest score band if boxes overlap excessively."""
        if not detections:
            return []
        sorted_dets = sorted(detections, key=lambda x: x['score'], reverse=True)
        keep = []
        
        for d in sorted_dets:
            box = d['box']
            overlap = False
            for k in keep:
                kbox = k['box']
                # Calculate IoU
                xi1 = max(box[0], kbox[0])
                yi1 = max(box[1], kbox[1])
                xi2 = min(box[2], kbox[2])
                yi2 = min(box[3], kbox[3])
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                kbox_area = (kbox[2] - kbox[0]) * (kbox[3] - kbox[1])
                union_area = box_area + kbox_area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                if iou > iou_thresh:
                    overlap = True
                    break
            if not overlap:
                keep.append(d)
                
        # Limit to top 5 bands (standard resistors have 4 or 5 bands)
        return keep[:5]

    def _draw_annotations(self, img: np.ndarray, calc_result: Dict[str, Any]) -> np.ndarray:
        """Renders bounding boxes, band order tags, and resistance banner onto image."""
        sorted_dets = calc_result.get('sorted_detections', [])
        h, w, _ = img.shape

        for i, det in enumerate(sorted_dets, start=1):
            box = [int(v) for v in det['box']]
            label = det['label']
            score = det['score']
            color_key = label.lower()
            bgr_color = COLOR_BGR_MAP.get(color_key, (0, 255, 0))

            # Draw rectangle
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), bgr_color, 2)

            # Draw label tag
            tag = f"#{i} {label} ({score:.0%})"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(box[1] - 6, th + 4)
            
            # Tag background
            cv2.rectangle(img, (box[0], ty - th - 2), (box[0] + tw + 6, ty + 2), (20, 20, 20), -1)
            cv2.rectangle(img, (box[0], ty - th - 2), (box[0] + tw + 6, ty + 2), bgr_color, 1)
            cv2.putText(img, tag, (box[0] + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw Resistance Summary Banner at bottom
        val_str = calc_result.get('formatted_value', 'Unknown')
        banner_text = f"Resistance: {val_str}"
        cv2.rectangle(img, (0, h - 32), (w, h), (15, 15, 15), -1)
        cv2.putText(img, banner_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA)
        
        return img

if __name__ == '__main__':
    import glob
    predictor = ResistorPredictor()
    test_images = glob.glob('resistor_images/test/images/*')
    if test_images:
        sample_img = test_images[0]
        print(f"Running prediction on sample: {sample_img}")
        res = predictor.predict(sample_img)
        print("Status:", res['status'])
        if res['status'] == 'success':
            calc = res['calculation']
            print(f"Detected Bands: {calc['bands']}")
            print(f"Calculated Resistance: {calc['formatted_value']}")
            print(f"Range: {calc['min_formatted']} - {calc['max_formatted']}")
