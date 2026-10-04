"""
Unit tests for Resistance Calculator and Resistor Band Detection Pipeline.
"""

import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from resistance_calculator import (
    calculate_4band_resistance,
    calculate_5band_resistance,
    format_resistance_value,
    order_and_calculate_from_detections
)

def test_4band_calculations():
    # 1. 10k Ohm +-5% (Brown, Black, Orange, Gold)
    res = calculate_4band_resistance('Brown', 'Black', 'Orange', 'Gold')
    assert res['nominal_ohms'] == 10000.0, f"Expected 10000, got {res['nominal_ohms']}"
    assert res['formatted_value'] == "10 kΩ ±5%", f"Expected '10 kΩ ±5%', got {res['formatted_value']}"
    assert res['min_ohms'] == 9500.0
    assert res['max_ohms'] == 10500.0
    print("[PASS] Test 1: 10k Ohm Resistor (Brown-Black-Orange-Gold)")

    # 2. 470 Ohm +-5% (Yellow, Violet, Brown, Gold)
    res2 = calculate_4band_resistance('Yellow', 'Violet', 'Brown', 'Side-Gold')
    assert res2['nominal_ohms'] == 470.0
    assert res2['formatted_value'] == "470 Ω ±5%"
    print("[PASS] Test 2: 470 Ohm Resistor (Yellow-Violet-Brown-Gold)")

    # 3. 1M Ohm +-10% (Brown, Black, Green, Silver)
    res3 = calculate_4band_resistance('Brown', 'Black', 'Green', 'Side-Silver')
    assert res3['nominal_ohms'] == 1000000.0
    assert res3['formatted_value'] == "1 MΩ ±10%"
    print("[PASS] Test 3: 1M Ohm Resistor (Brown-Black-Green-Silver)")

    # 4. 2.2 Ohm +-5% (Red, Red, Gold, Gold)
    res4 = calculate_4band_resistance('Red', 'Red', 'Gold', 'Gold')
    assert abs(res4['nominal_ohms'] - 2.2) < 1e-4
    assert res4['formatted_value'] == "2.2 Ω ±5%"
    print("[PASS] Test 4: 2.2 Ohm Resistor (Red-Red-Gold-Gold)")

def test_5band_calculations():
    # 1k Ohm +-1% (Brown, Black, Black, Brown, Brown)
    res = calculate_5band_resistance('Brown', 'Black', 'Black', 'Brown', 'Brown')
    assert res['nominal_ohms'] == 1000.0
    assert res['formatted_value'] == "1 kΩ ±1%"
    print("[PASS] Test 5: 5-Band 1k Ohm Precision Resistor")

def test_orientation_auto_correction():
    # Given detections ordered backwards (Gold, Orange, Black, Brown) -> should flip to (Brown, Black, Orange, Gold)
    backwards_detections = [
        {'label': 'Side-Gold', 'box': [10, 10, 20, 50], 'score': 0.95},
        {'label': 'Orange',    'box': [40, 10, 50, 50], 'score': 0.92},
        {'label': 'Black',     'box': [70, 10, 80, 50], 'score': 0.90},
        {'label': 'Brown',     'box': [100, 10, 110, 50], 'score': 0.94},
    ]
    res = order_and_calculate_from_detections(backwards_detections)
    assert res['reversed_order'] == True, "Orientation should have been flipped"
    assert res['nominal_ohms'] == 10000.0, f"Expected 10000, got {res['nominal_ohms']}"
    assert res['bands'] == ['Brown', 'Black', 'Orange', 'Side-Gold']
    print("[PASS] Test 6: Orientation Auto-Flip (Backwards Tolerance Band Detected & Corrected)")

if __name__ == '__main__':
    test_4band_calculations()
    test_5band_calculations()
    test_orientation_auto_correction()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
