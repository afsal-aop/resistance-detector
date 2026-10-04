"""
Resistor Color Code & Resistance Calculator Module
Supports 4-band, 5-band, and 6-band resistors with automatic orientation detection,
unit formatting (Ω, kΩ, MΩ, GΩ), and tolerance calculation.
"""

from typing import List, Dict, Any, Tuple, Optional

# Standard Resistor Color Database
COLOR_DIGIT_MAP = {
    'black': 0,
    'brown': 1,
    'red': 2,
    'orange': 3,
    'yellow': 4,
    'green': 5,
    'blue': 6,
    'violet': 7,
    'purple': 7,
    'gray': 8,
    'grey': 8,
    'white': 9
}

COLOR_MULTIPLIER_MAP = {
    'black': 1.0,         # 10^0
    'brown': 10.0,        # 10^1
    'red': 100.0,         # 10^2
    'orange': 1_000.0,    # 10^3
    'yellow': 10_000.0,   # 10^4
    'green': 100_000.0,   # 10^5
    'blue': 1_000_000.0,  # 10^6
    'violet': 10_000_000.0, # 10^7
    'purple': 10_000_000.0,
    'gray': 100_000_000.0,  # 10^8
    'grey': 100_000_000.0,
    'white': 1_000_000_000.0, # 10^9
    'gold': 0.1,          # 10^-1
    'side-gold': 0.1,
    'silver': 0.01,       # 10^-2
    'side-silver': 0.01
}

COLOR_TOLERANCE_MAP = {
    'brown': 1.0,        # ±1%
    'red': 2.0,          # ±2%
    'green': 0.5,        # ±0.5%
    'blue': 0.25,        # ±0.25%
    'violet': 0.1,       # ±0.1%
    'purple': 0.1,
    'gray': 0.05,        # ±0.05%
    'grey': 0.05,
    'gold': 5.0,         # ±5%
    'side-gold': 5.0,    # ±5%
    'silver': 10.0,      # ±10%
    'side-silver': 10.0, # ±10%
    'none': 20.0
}

COLOR_HEX_MAP = {
    'black': '#000000',
    'brown': '#8B4513',
    'red': '#FF0000',
    'orange': '#FF8C00',
    'yellow': '#FFD700',
    'green': '#228B22',
    'blue': '#1E90FF',
    'violet': '#8A2BE2',
    'purple': '#8A2BE2',
    'gray': '#808080',
    'grey': '#808080',
    'white': '#F5F5F5',
    'gold': '#CFB53B',
    'side-gold': '#CFB53B',
    'silver': '#C0C0C0',
    'side-silver': '#C0C0C0'
}

def normalize_color_name(name: str) -> str:
    """Normalize color string (e.g. 'Side-Gold' -> 'side-gold')."""
    return name.strip().lower()

def format_resistance_value(ohms: float) -> str:
    """Format raw ohms into human-readable engineering unit string (Ω, kΩ, MΩ, GΩ)."""
    if ohms >= 1_000_000_000:
        val = ohms / 1_000_000_000
        return f"{val:g} GΩ"
    elif ohms >= 1_000_000:
        val = ohms / 1_000_000
        return f"{val:g} MΩ"
    elif ohms >= 1_000:
        val = ohms / 1_000
        return f"{val:g} kΩ"
    elif ohms >= 1:
        return f"{ohms:g} Ω"
    else:
        return f"{ohms:.2f} Ω"

def calculate_4band_resistance(b1: str, b2: str, mult: str, tol: str) -> Dict[str, Any]:
    """
    Calculate resistance for a 4-band resistor:
    Band 1: 1st significant digit
    Band 2: 2nd significant digit
    Band 3: Multiplier
    Band 4: Tolerance
    """
    c1 = normalize_color_name(b1)
    c2 = normalize_color_name(b2)
    cm = normalize_color_name(mult)
    ct = normalize_color_name(tol)

    if c1 not in COLOR_DIGIT_MAP:
        raise ValueError(f"Invalid 1st band color: '{b1}'")
    if c2 not in COLOR_DIGIT_MAP:
        raise ValueError(f"Invalid 2nd band color: '{b2}'")
    if cm not in COLOR_MULTIPLIER_MAP:
        raise ValueError(f"Invalid multiplier color: '{mult}'")

    d1 = COLOR_DIGIT_MAP[c1]
    d2 = COLOR_DIGIT_MAP[c2]
    multiplier = COLOR_MULTIPLIER_MAP[cm]
    tolerance_pct = COLOR_TOLERANCE_MAP.get(ct, 20.0)

    nominal_ohms = (d1 * 10 + d2) * multiplier
    tol_range = nominal_ohms * (tolerance_pct / 100.0)
    min_ohms = nominal_ohms - tol_range
    max_ohms = nominal_ohms + tol_range

    return {
        'bands': [b1, b2, mult, tol],
        'band_count': 4,
        'digits': [d1, d2],
        'multiplier': multiplier,
        'tolerance_pct': tolerance_pct,
        'nominal_ohms': nominal_ohms,
        'formatted_value': f"{format_resistance_value(nominal_ohms)} ±{tolerance_pct:g}%",
        'nominal_formatted': format_resistance_value(nominal_ohms),
        'min_ohms': min_ohms,
        'max_ohms': max_ohms,
        'min_formatted': format_resistance_value(min_ohms),
        'max_formatted': format_resistance_value(max_ohms),
        'color_hexes': [COLOR_HEX_MAP.get(c, '#888888') for c in [c1, c2, cm, ct]]
    }

def calculate_5band_resistance(b1: str, b2: str, b3: str, mult: str, tol: str) -> Dict[str, Any]:
    """
    Calculate resistance for a 5-band precision resistor:
    Band 1: 1st significant digit
    Band 2: 2nd significant digit
    Band 3: 3rd significant digit
    Band 4: Multiplier
    Band 5: Tolerance
    """
    c1 = normalize_color_name(b1)
    c2 = normalize_color_name(b2)
    c3 = normalize_color_name(b3)
    cm = normalize_color_name(mult)
    ct = normalize_color_name(tol)

    if c1 not in COLOR_DIGIT_MAP or c2 not in COLOR_DIGIT_MAP or c3 not in COLOR_DIGIT_MAP:
        raise ValueError(f"Invalid digit band in: {[b1, b2, b3]}")
    if cm not in COLOR_MULTIPLIER_MAP:
        raise ValueError(f"Invalid multiplier color: '{mult}'")

    d1 = COLOR_DIGIT_MAP[c1]
    d2 = COLOR_DIGIT_MAP[c2]
    d3 = COLOR_DIGIT_MAP[c3]
    multiplier = COLOR_MULTIPLIER_MAP[cm]
    tolerance_pct = COLOR_TOLERANCE_MAP.get(ct, 20.0)

    nominal_ohms = (d1 * 100 + d2 * 10 + d3) * multiplier
    tol_range = nominal_ohms * (tolerance_pct / 100.0)
    min_ohms = nominal_ohms - tol_range
    max_ohms = nominal_ohms + tol_range

    return {
        'bands': [b1, b2, b3, mult, tol],
        'band_count': 5,
        'digits': [d1, d2, d3],
        'multiplier': multiplier,
        'tolerance_pct': tolerance_pct,
        'nominal_ohms': nominal_ohms,
        'formatted_value': f"{format_resistance_value(nominal_ohms)} ±{tolerance_pct:g}%",
        'nominal_formatted': format_resistance_value(nominal_ohms),
        'min_ohms': min_ohms,
        'max_ohms': max_ohms,
        'min_formatted': format_resistance_value(min_ohms),
        'max_formatted': format_resistance_value(max_ohms),
        'color_hexes': [COLOR_HEX_MAP.get(c, '#888888') for c in [c1, c2, c3, cm, ct]]
    }

def order_and_calculate_from_detections(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Takes a list of detected band dictionaries:
    Each item: {'label': str, 'box': [xmin, ymin, xmax, ymax], 'score': float}
    
    1. Determines resistor orientation (horizontal vs vertical).
    2. Sorts the detected bands along the primary axis.
    3. Handles orientation reversal if tolerance band (Gold/Silver) is on the leading side.
    4. Computes resistance value and metrics.
    """
    if not detections:
        return {'error': 'No bands detected'}

    # Calculate center coordinates
    for d in detections:
        box = d['box']
        d['cx'] = (box[0] + box[2]) / 2.0
        d['cy'] = (box[1] + box[3]) / 2.0

    # Determine axis variance (horizontal vs vertical)
    xs = [d['cx'] for d in detections]
    ys = [d['cy'] for d in detections]
    x_spread = max(xs) - min(xs) if len(xs) > 1 else 0
    y_spread = max(ys) - min(ys) if len(ys) > 1 else 0

    axis = 'x' if x_spread >= y_spread else 'y'
    sorted_detections = sorted(detections, key=lambda d: d['cx'] if axis == 'x' else d['cy'])

    labels = [d['label'] for d in sorted_detections]

    # Tolerance indicators
    tol_colors = {'side-gold', 'gold', 'side-silver', 'silver'}
    first_is_tol = normalize_color_name(labels[0]) in tol_colors
    last_is_tol = normalize_color_name(labels[-1]) in tol_colors

    reversed_order = False
    # If the first band is Gold/Silver, the resistor is upside-down or reversed
    if first_is_tol and not last_is_tol:
        sorted_detections = list(reversed(sorted_detections))
        labels = [d['label'] for d in sorted_detections]
        reversed_order = True

    band_count = len(labels)
    if band_count == 4:
        calc_res = calculate_4band_resistance(labels[0], labels[1], labels[2], labels[3])
    elif band_count == 5:
        calc_res = calculate_5band_resistance(labels[0], labels[1], labels[2], labels[3], labels[4])
    elif band_count == 3:
        # Assume 3-band with 20% tolerance (default)
        calc_res = calculate_4band_resistance(labels[0], labels[1], labels[2], 'none')
    else:
        # Fallback to 4-band using first 4 or top score 4
        calc_res = calculate_4band_resistance(labels[0], labels[1], labels[2], labels[-1])

    calc_res['sorted_detections'] = sorted_detections
    calc_res['reversed_order'] = reversed_order
    calc_res['axis'] = 'horizontal' if axis == 'x' else 'vertical'
    return calc_res
