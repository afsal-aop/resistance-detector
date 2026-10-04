# ⚡ Resistor Color Band AI Detector & Resistance Calculator

An end-to-end Computer Vision & Deep Learning system built with **MobileNet CNN / Faster R-CNN (PyTorch)** to automatically detect color bands on resistor images and calculate their nominal resistance value, tolerance, and operating range.

---

## 🚀 Features

- **MobileNet Deep Learning Detector**: Detects and localizes all resistor color bands (`Black`, `Brown`, `Red`, `Orange`, `Yellow`, `Green`, `Blue`, `Side-Gold`, `Side-Silver`).
- **Smart Spatial Ordering**: Dynamically projects detected band bounding boxes along the primary orientation axis (horizontal or vertical).
- **Auto-Orientation Correction**: Detects tolerance bands (`Side-Gold` / `Side-Silver`) to automatically resolve whether the resistor is oriented left-to-right or right-to-left.
- **Precision Resistance Engine**:
  - Supports standard **4-band**, **5-band**, and **6-band** resistor color codes (IEC 60062 / EIA standards).
  - Engineering unit formatting ($\Omega$, $k\Omega$, $M\Omega$, $G\Omega$).
  - Tolerance percentage ($\pm\%$) and exact minimum/maximum resistance bounds.
- **Interactive Streamlit Web UI**:
  - Drag-and-drop image upload or choose from test dataset samples.
  - Real-time bounding box visualization with confidence scores and color swatches.
  - One-click orientation flip option.
  - Interactive manual color code calculator with real-time resistor preview.

---

## 📁 Project Structure

```
with_other_antigravity/
├── resistor_images/               # Dataset (train/valid/test images & YOLO labels)
│   ├── train/
│   ├── valid/
│   └── test/
├── resistance_calculator.py       # Core resistance calculation & orientation logic
├── dataset.py                     # PyTorch Dataset loader & augmentation pipeline
├── model.py                       # MobileNet Faster R-CNN detector architecture
├── train.py                       # Model training loop & validation evaluator
├── predict.py                     # Single-image inference & visualization module
├── app.py                         # Streamlit interactive web application
├── test_calculator.py             # Unit test suite
└── weights/
    └── resistor_detector.pth      # Trained MobileNet checkpoint
```

---

## 🛠️ Installation & Requirements

Ensure the following packages are installed in your Python environment:

```bash
pip install torch torchvision opencv-python pillow streamlit matplotlib
```

---

## 🏃 Quick Start

### 1. Launch the Interactive Web App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test the AI scanner and manual calculator.

### 2. Run Single Image Prediction from Python / CLI
```bash
python predict.py
```
Or in your own Python script:
```python
from predict import ResistorPredictor

predictor = ResistorPredictor(weights_path='weights/resistor_detector.pth')
result = predictor.predict('resistor_images/test/images/resbody-100_jpg.rf.xxx.jpg')

if result['status'] == 'success':
    calc = result['calculation']
    print("Detected Bands:", calc['bands'])
    print("Calculated Resistance:", calc['formatted_value'])
    print("Operating Range:", calc['min_formatted'], "to", calc['max_formatted'])
```

### 3. Calculate Resistance Directly from Color Names
```python
from resistance_calculator import calculate_4band_resistance, calculate_5band_resistance

# 4-Band Resistor: Yellow (4), Violet (7), Red (x100), Gold (+-5%)
result = calculate_4band_resistance('Yellow', 'Violet', 'Red', 'Gold')
print(result['formatted_value'])
# Output: 4.7 kΩ ±5%
```

### 4. Train / Fine-tune the Model
```bash
python train.py 6  # Trains for 6 epochs
```

---

## 📊 Standard Resistor Color Code Table

| Band Color | Digit | Multiplier | Tolerance |
|---|---|---|---|
| **Black** | 0 | $10^0$ (1) | — |
| **Brown** | 1 | $10^1$ (10) | $\pm 1\%$ |
| **Red** | 2 | $10^2$ (100) | $\pm 2\%$ |
| **Orange** | 3 | $10^3$ (1 k$\Omega$) | — |
| **Yellow** | 4 | $10^4$ (10 k$\Omega$) | — |
| **Green** | 5 | $10^5$ (100 k$\Omega$) | $\pm 0.5\%$ |
| **Blue** | 6 | $10^6$ (1 M$\Omega$) | $\pm 0.25\%$ |
| **Violet** | 7 | $10^7$ (10 M$\Omega$) | $\pm 0.1\%$ |
| **Gray** | 8 | $10^8$ (100 M$\Omega$) | $\pm 0.05\%$ |
| **White** | 9 | $10^9$ (1 G$\Omega$) | — |
| **Side-Gold** | — | $10^{-1}$ (0.1) | $\pm 5\%$ |
| **Side-Silver** | — | $10^{-2}$ (0.01) | $\pm 10\%$ |

---

## 🧪 Testing

Run the automated test suite to verify formulas and orientation handling:
```bash
python test_calculator.py
```
