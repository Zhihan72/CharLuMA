import os
import json
import math
import pandas as pd
from copy import deepcopy
from collections import defaultdict, Counter, OrderedDict
import matplotlib.pyplot as plt
import matplotlib as mpl

import sys
sys.path.append(os.path.dirname(__file__))

_VALID_VA_PY  = {"top", "bottom", "center", "baseline", "center_baseline"}
_VALID_HA_PY  = {"left", "right", "center"}
_VALID_VA_R   = {"top", "bottom", "center", "middle"}
_VALID_HA_R   = {"left", "right", "center", "middle"}

def _to_float(x):
    try:
        if isinstance(x, str):
            x = x.strip()
        return float(x)
    except Exception:
        return None

def _norm_str(x):
    if x is None:
        return None
    s = str(x).strip().lower()

    if s in {"centre", "middle"}:
        return "center"
    if s in {"center_baseline", "centerbaseline"}:
        return "center_baseline"
    return s

def _coerce_va_for_target(s, trans_lang):
    if trans_lang == "python":
        return s if s in _VALID_VA_PY else "center"
    else:
        return s if s in _VALID_VA_R else "center"

def _coerce_ha_for_target(s, trans_lang):
    if trans_lang == "python":
        return s if s in _VALID_HA_PY else "center"
    else:
        return s if s in _VALID_HA_R else "center"

def map_annot_vjust(vjust, orig_lang, trans_lang):
    s = _norm_str(vjust)
    if s in {"top", "bottom", "center", "baseline", "center_baseline"}:
        if trans_lang == "r" and s in {"baseline", "center_baseline"}:
            s = "center"
        return _coerce_va_for_target(s, trans_lang)
    
    f = _to_float(vjust)
    if f is not None:
        if f <= -0.25:
            return _coerce_va_for_target("bottom", trans_lang)
        if f >= 0.25:
            return _coerce_va_for_target("top", trans_lang)
        return _coerce_va_for_target("center", trans_lang)
    
    return _coerce_va_for_target("center", trans_lang)


def map_annot_hjust(hjust, orig_lang, trans_lang):
    s = _norm_str(hjust)
    if s in {"left", "right", "center"}:
        return _coerce_ha_for_target(s, trans_lang)
    f = _to_float(hjust)
    if f is not None:
        if f <= 0.25:
            return _coerce_ha_for_target("left", trans_lang)
        if f >= 0.75:
            return _coerce_ha_for_target("right", trans_lang)
        return _coerce_ha_for_target("center", trans_lang)
    return _coerce_ha_for_target("center", trans_lang)

matplotlib_legend_map = {
    0: "best",
    1: "upper right",
    2: "upper left",
    3: "lower left",
    4: "lower right",
    5: "right",
    6: "center left",
    7: "center right",
    8: "lower center",
    9: "upper center",
    10: "center"
}

py_to_r = {
    "best": "right",
    "upper right": "right",
    "lower right": "right",
    "upper left": "left",
    "lower left": "left",
    "center right": "right",
    "center left": "left",
    "upper center": "top",
    "lower center": "bottom",
    "center": "top"
}

py_to_latex = {
    "best": "best",
    "upper right": "north east",
    "upper left": "north west",
    "lower left": "south west",
    "lower right": "south east",
    "right": "east",
    "left": "west",
    "center right": "east",
    "center left": "west",
    "lower center": "south",
    "upper center": "north",
    "center": "center"
}

r_to_py = {
    "right": "center right",
    "left": "center left",
    "top": "upper center",
    "bottom": "lower center",
    "center": "center",
    "none": "none"
}


def map_legend(loc, ncol, orig_lang, trans_lang):
    ncol = ncol or 1
    defaults = {"python": "best", "r": "right", "latex": "north east"}
    if loc is None:
        loc = defaults.get(orig_lang, "best")
    elif isinstance(loc, int) or (isinstance(loc, str) and loc.strip().isdigit()):
        loc = matplotlib_legend_map.get(int(loc), "best")
    loc = str(loc).strip().lower()
    mappings = {
        ("python", "r"): py_to_r,
        ("python", "latex"): py_to_latex,
        ("r", "python"): r_to_py
    }

    if (orig_lang, trans_lang) in mappings:
        mapped_loc = mappings[(orig_lang, trans_lang)].get(loc, defaults.get(trans_lang))
    elif orig_lang == "r" and trans_lang == "latex":
        mapped_loc = py_to_latex.get(r_to_py.get(loc, "best"), defaults["latex"])
    else:
        mapped_loc = loc or defaults.get(trans_lang)

    return mapped_loc, ncol

def map_font(style_string, trans_lang):
    if not style_string:
        style_string = ""
    style_string = style_string.lower().strip()

    is_bold = "bold" in style_string
    is_italic = "italic" in style_string

    if trans_lang == "python":
        weight = "bold" if is_bold else "normal"
        style = "italic" if is_italic else "normal"
        fontface = ""

    elif trans_lang == "r":
        weight = ""
        style = ""
        if is_bold and is_italic:
            fontface = "bold.italic"
        elif is_bold:
            fontface = "bold"
        elif is_italic:
            fontface = "italic"
        else:
            fontface = "plain"

    elif trans_lang == "latex":
        weight = ""
        style = ""
        faces = []
        if is_bold:
            faces.append("\\bfseries")
        if is_italic:
            faces.append("\\itshape")
        fontface = "".join(faces)

    else:
        raise ValueError(f"Unsupported trans_lang: {trans_lang}")

    return weight, style, fontface

def map_font_size(size, trans_lang):
    if trans_lang == "latex":
        if not size or isinstance(size, str):
            return "\\normalsize"
        if size <= 8:
            return "\\scriptsize"
        elif size <= 10:
            return "\\footnotesize"
        elif size <= 12:
            return "\\small"
        elif size <= 14:
            return "\\normalsize"
        elif size <= 16:
            return "\\large"
        elif size <= 18:
            return "\\Large"
        else:
            return "\\Huge"
    elif trans_lang in {"python", "r"}:
        return size if size not in [None, "None", "NA"] else 12 
    else:
        raise ValueError(f"Unsupported target language: {trans_lang}")

def cmap_to_ggplot_or_colors(cmap_name):
    if not isinstance(cmap_name, str) or not cmap_name.strip():
        raise ValueError("cmap_name must be a non-empty string.")

    cmap_lc = cmap_name.strip().lower()
    viridis_opts = {"magma", "inferno", "viridis", "cividis", "plasma"}
    brewer_map = {
        "coolwarm": "RdBu",
        "spectral": "Spectral",
        "rdylbu":   "RdYlBu",
        "rdylgn":   "RdYlGn",
        "rdbu":     "RdBu",
        "purd":     "PuRd",
        "pugn":     "PuGn",
        "brbg":     "BrBG",
        "puor":     "PuOr",
        "set1":     "Set1",
        "set2":     "Set2",
        "set3":     "Set3",
        "paired":   "Paired",
        "accent":   "Accent",
    }
    if cmap_lc in viridis_opts:
        return (cmap_lc, None, None, None)
    if cmap_lc in brewer_map:
        return (brewer_map[cmap_lc], None, None, None)
    
    try:
        cmap = plt.get_cmap(cmap_name)
    except ValueError as e:
        raise ValueError(f"Unknown Matplotlib colormap: {cmap_name}") from e

    pts = [0.0, 0.5, 1.0]
    low, mid, high = [mpl.colors.to_hex(cmap(p)) for p in pts]
    return (None, low, mid, high)

r_line_numeric_to_logical = {
    1: "solid",
    2: "dashed",
    3: "dotted",
    4: "dotdash",
    5: "longdash",
    6: "twodash"
}

line_style_map = {
    "solid":    {"python": "-", "r": "solid", "latex": "solid"},
    "dashed":   {"python": "--", "r": "dashed", "latex": "dashed"},
    "dotted":   {"python": ":", "r": "dotted", "latex": "dotted"},
    "dotdash":  {"python": "-.", "r": "dotdash", "latex": "dashdotted"},
    "longdash": {"python": "--", "r": "longdash", "latex": "dashed"},
    "twodash":  {"python": "--", "r": "twodash", "latex": "dashed"},
}

logical_size_tiers = {
    "tiny":     {"python": 1.0, "r": 0.5, "latex": 0.2},
    "small":    {"python": 1.5, "r": 0.8, "latex": 0.4},
    "medium":   {"python": 2.0, "r": 1.0, "latex": 0.6},
    "large":    {"python": 2.5, "r": 1.5, "latex": 0.8},
    "xlarge":   {"python": 3.0, "r": 2.0, "latex": 1.0},
    "xxlarge":  {"python": 4.0, "r": 3.0, "latex": 1.5},
}

r_marker_numeric_to_logical = {
    0:  "square",
    1:  "circle",
    2:  "triangle",
    3:  "plus",
    4:  "cross",
    5:  "diamond",
    6:  "triangle down",
    7:  "square",
    8:  "star",
    9:  "diamond",
    10: "circle",
    11: "triangle",
    12: "square",
    13: "circle",
    14: "square",
    15: "square",
    16: "circle",
    17: "triangle",
    18: "diamond",
    19: "circle",
    20: "circle",
    21: "circle",
    22: "square",
    23: "diamond",
    24: "triangle",
    25: "triangle"
}


marker_style_map={
    "circle":{"python":"o","r":"circle","latex":"o"},
    "triangle":{"python":"^","r":"triangle","latex":"triangle*"},
    "triangle down":{"python":"v","r":"triangle","latex":"triangle*"},
    "triangle left":{"python":"<","r":"triangle","latex":"triangle*"},
    "triangle right":{"python":">","r":"triangle","latex":"triangle*"},
    "square":{"python":"s","r":"square","latex":"square*"},
    "star":{"python":"*","r":"star","latex":"asterisk"},
    "plus":{"python":"+","r":"plus","latex":"asterisk"},
    "cross":{"python":"x","r":"cross","latex":"asterisk"},
    "diamond":{"python":"D","r":"diamond","latex":"diamond*"},
    "pentagon":{"python":"p","r":"square","latex":"pentagon*"},
    "hline": {"python":"_","r":"square","latex":"-"}
}


logical_marker_size = {
    "tiny":     {"python": 4, "r": 1, "latex": "1pt"},
    "small":    {"python": 6, "r": 2, "latex": "2pt"},
    "medium":   {"python": 8, "r": 3, "latex": "3pt"},
    "large":    {"python": 10, "r": 4, "latex": "4pt"},
    "xlarge":   {"python": 12, "r": 5, "latex": "5pt"},
    "xxlarge":  {"python": 14, "r": 6, "latex": "6pt"},
}

def normalize_r_style(style, style_type):
    if isinstance(style, (int, float)):
        if style_type == "line":
            return r_line_numeric_to_logical.get(style)
        elif style_type == "marker":
            return r_marker_numeric_to_logical.get(style)
    if isinstance(style, str) and style.isdigit() and len(style) % 2 == 0:
        hex_dash_map = {
            "22": "dashed",
            "3313": "dotdash",
            "42": "longdash",
        }
        return hex_dash_map.get(style, "solid")
    return style

def map_style(style, src_lang, tgt_lang, style_map, style_type=None):
    if src_lang == "r":
        norm = normalize_r_style(style, style_type)
        if norm is None:
            print(f"Warning: normalize_r_style returned None for {style!r}")
        else:
            style = norm
    for logical, mapping in style_map.items():
        if mapping.get(src_lang) == style:
            tgt = mapping.get(tgt_lang)
            if tgt is None:
                print(f"Warning: found tier={logical!r} but no {tgt_lang!r} entry")
                return style
            return tgt
    print(f"Warning: no mapping for style={style!r} from {src_lang!r} to {tgt_lang!r}")
    return style

def map_size(value, src_lang, tgt_lang, size_map, tol=0.6):
    if isinstance(value, list):
        print(f"Warning: got list {value!r}, returning 1")
        return 1
    if value is None or not isinstance(value, (int, float)):
        print(f"Warning: unexpected size value {value!r}")
        return value
    for logical, vals in size_map.items():
        src = vals.get(src_lang)
        tgt = vals.get(tgt_lang)
        if not isinstance(src, (int, float)):
            print(f"Warning: no numeric mapping for src_lang={src_lang!r} in tier={logical!r}")
            continue
        if abs(src - value) < tol:
            if tgt is None:
                print(f"Warning: matched tier={logical!r} but no tgt_lang mapping for {tgt_lang!r}")
                return value
            return tgt
    print(f"Warning: no mapping for value={value!r} from {src_lang!r} to {tgt_lang!r}")
    return value

def map_line_style(style, src_lang, tgt_lang):
    return map_style(style, src_lang, tgt_lang, line_style_map, style_type="line")

def map_line_width(size, src_lang, tgt_lang):
    return map_size(size, src_lang, tgt_lang, logical_size_tiers)

def map_marker_style(style, src_lang, tgt_lang):
    return map_style(style, src_lang, tgt_lang, marker_style_map, style_type="marker")

def map_marker_size(size, src_lang, tgt_lang):
    return map_size(size, src_lang, tgt_lang, logical_marker_size)