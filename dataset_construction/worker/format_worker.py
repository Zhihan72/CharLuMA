import os
import re
import json
import pandas as pd
import numpy as np
from copy import deepcopy
from collections import defaultdict, Counter, OrderedDict
from matplotlib.colors import to_hex, is_color_like, to_rgb
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(__file__))

def extract_sorted_ticks(ticks, axis):
    if not ticks:
        return [], []

    index = 0 if axis == "x" else 1

    try:
        sorted_items = sorted(
            [item for item in ticks if item.get("text") is not None],
            key=lambda x: float(x["position"][index])
        )
        labels = [item["text"] for item in sorted_items]
        positions = [float(item["position"][index]) for item in sorted_items]
    except (TypeError, ValueError, IndexError, KeyError):
        labels = [item["text"] for item in ticks if item.get("text") is not None]
        positions = [None] * len(labels)

    return labels, positions

def rgb_percent_to_hex(color):
    if color is None:
        return None

    def normalize_named_color(c):
        if isinstance(c, str):
            return c.lower().replace(" ", "")
        return c

    def resolve_c_code(c):
        if isinstance(c, str) and re.fullmatch(r"c[0-9]", c.lower()):
            index = int(c[1])
            cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
            if 0 <= index < len(cycle):
                return cycle[index]
            else:
                raise ValueError(f"Color cycle index out of range: {c}")
        return c

    def resolve_shaded_named_color(c):
        if not isinstance(c, str):
            return c

        match = re.fullmatch(r"([a-z]+)(\d{1,3})", c)
        if match:
            base_name, percent_str = match.groups()
            percent = int(percent_str)
            if not (0 <= percent <= 100):
                raise ValueError(f"Invalid shade percentage: {c}")
            try:
                base_rgb = to_rgb(base_name)
            except ValueError:
                raise ValueError(f"Unknown base color: {base_name}")
            white_rgb = (1.0, 1.0, 1.0)
            blend = tuple(
                (1 - percent / 100) * w + (percent / 100) * b
                for w, b in zip(white_rgb, base_rgb)
            )
            return blend
        return c

    def convert_single(c):
        try:
            c_norm = normalize_named_color(c)
            c_resolved = resolve_c_code(c_norm)
            c_shaded = resolve_shaded_named_color(c_resolved)
            return to_hex(c_shaded)
        except ValueError:
            raise ValueError(f"Invalid color: {c}")

    if isinstance(color, (str, tuple, list)):
        if isinstance(color, (list, tuple)) and all(isinstance(x, (str, tuple, list)) for x in color):
            return [convert_single(c) for c in color]
        return convert_single(color)

    raise ValueError(f"Invalid color format: {color}")

def hex_to_rgb(hex_code: str) -> tuple[int, int, int]:
    hex_code = hex_code.lstrip('#')
    if len(hex_code) != 6:
        raise ValueError(f"Expected 6‑digit hex, got {hex_code!r}")
    r = int(hex_code[0:2], 16)
    g = int(hex_code[2:4], 16)
    b = int(hex_code[4:6], 16)
    return [r, g, b]

def escape_string(s, trans_lang):
    if s is None:
        return ""
    if trans_lang in ['python', 'r']:
        return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    if trans_lang =='latex':
        s = str(s)
        return (
            s.replace("\\", "\\\\")
            .replace("_", "\\_")
            .replace("$", "\\$")
            .replace("&", "\\&")
            .replace("%", "\\%")
            .replace("#", "\\#")
            .replace("^", "\\^{}")
            .replace("~", "\\~{}")
            .replace("{", "\\{")
            .replace("}", "\\}")
            .replace("[", "")
            .replace("]", "")
            .replace("\n", " ")
        )

def is_number(s):
    try:
        float(str(s).strip())
        return True
    except ValueError:
        return False

def is_all_none(data):
    if data==[]:
        return True
    if isinstance(data[0], list):
        return all(x in [None, "None"] for row in data for x in row)
    else:
        return all(x in [None, "None"] for x in data)

def is_numeric(s):
    try:
        float(s.replace("−", "-"))
        return True
    except Exception:
        return False

def is_numeric_tick_list(ticks):
    for tick in ticks:
        if tick is None:
            continue
        try:
            float(tick)
        except (ValueError, TypeError):
            return False
    return True

def format_list_for_lang(py_list, trans_lang, value_type="str"):
    def to_r_str_vector(lst):
        if all(isinstance(sublist, list) for sublist in lst):
            return "list(" + ", ".join(
                "c(" + ", ".join(f"'{str(item)}'" for item in sublist) + ")" for sublist in lst
            ) + ")"
        return "c(" + ", ".join(f"'{str(item)}'" for item in lst) + ")"

    def to_r_num_vector(lst):
        if all(isinstance(sublist, list) for sublist in lst):
            return "list(" + ", ".join(
                "c(" + ", ".join(str(item) for item in sublist) + ")" for sublist in lst
            ) + ")"
        return "c(" + ", ".join(str(item) for item in lst) + ")"

    def to_latex_brace_list(lst):
        return "{" + ", ".join(escape_string(str(x), "latex") for x in lst) + "}"

    def to_latex_str_list(label_list):
        lines = ["{"]
        for label in label_list:
            clean_label = label.strip().strip("{}") if isinstance(label, str)  else label
            escaped = escape_string(clean_label, "latex")
            lines.append(f"    {{{escaped}}},")
        lines.append("}")
        return "\n".join(lines)

    if trans_lang == "python":
        return repr(py_list)
    elif trans_lang == "r":
        return to_r_num_vector(py_list) if value_type == "num" else to_r_str_vector(py_list)
    elif trans_lang == "latex":
        return to_latex_brace_list(py_list) if value_type == "num" else to_latex_str_list(py_list)
    else:
        raise ValueError(f"Unsupported target language: {trans_lang}")

def subplot_positions_py(n_row, n_col):
    if n_row == 1:
        return [str(c) for c in range(n_col)]
    if n_col == 1:
        return [str(c) for c in range(n_row)]
    return [f"{r},{c}" for r in range(n_row) for c in range(n_col)]

def generate_latex_color_define(colors, background_color, key='ax_0"'):
    def flatten_colors(data):
        flat = []
        if isinstance(data, list):
            for item in data:
                flat.extend(flatten_colors(item))
        elif isinstance(data, str) and data.strip():
            flat.append(data)
        return flat

    def map_structure(data, color_map):
        if isinstance(data, list):
            return [map_structure(item, color_map) for item in data]
        elif isinstance(data, str) and data.strip():
            return color_map.get(data, '')
        else:
            return []
    
    key_num = key.split("_")[-1]
    flat_colors = flatten_colors(colors)
    unique_colors = list(OrderedDict.fromkeys(flat_colors))
    unique_labels = [f"c{key_num}{i}" for i in range(len(unique_colors))]
    color_map = dict(zip(unique_colors, unique_labels))
    color_labels = map_structure(colors, color_map)
    color_define_lines = [
        "\\definecolor{{{}}}{{HTML}}{{{}}}".format(label, color.replace('#', '').upper())
        for label, color in zip(unique_labels, unique_colors)
    ]
    color_define_lines.append(
        "\\definecolor{{cb}}{{HTML}}{{{}}}".format(background_color.replace('#', '').upper())
    )

    return '\n'.join(color_define_lines), color_labels


def find_nearest_tick_label(patch_tick_pos, tick_labels, tick_pos):
    if not tick_labels or not tick_pos:
        return None
    idx = min(range(len(tick_pos)), key=lambda i: abs(patch_tick_pos - tick_pos[i]))
    return tick_labels[idx]


def find_label_by_color(patch_color, label_to_color):
    if isinstance(label_to_color, dict):
        for label, color in label_to_color.items():
            color = rgb_percent_to_hex(color)
            if color.lower() == patch_color.lower():
                return label
    return None


def get_sublist_by_indices(data, indices):

    return [data[i] for i in indices]

def normalize_vertices(vertices):
    vertices = vertices - vertices.mean(axis=0)
    max_norm = np.linalg.norm(vertices, axis=1).max()
    return vertices / max_norm if max_norm > 0 else vertices


def split_polygon_fill(x, y):
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    n = len(x)
    if n < 5:
        raise ValueError("Too few points to split meaningfully")

    mid = n // 2
    keep_indices = [i for i in range(n) if i not in {0, mid, n - 1}]
    
    x_filtered = [x[i] for i in keep_indices]
    y_filtered = [y[i] for i in keep_indices]
    
    half = len(x_filtered) // 2
    x1 = x_filtered[:half]
    y1 = y_filtered[:half]
    y2 = list(reversed(y_filtered[half:]))

    return x1, y1, y2


def is_between(y, y_low, y_high):
    if not (len(y) == len(y_low) == len(y_high)):
        print("Error in is_between func: All input lists must have the same length!")
        return False

    if is_all_none(y) or is_all_none(y_low) or is_all_none(y_high):
        print("Error in is_between func: The input list in in_between function may be empty!")
        return False
    
    return all(low < val < high for val, low, high in zip(y, y_low, y_high))


def py_str_to_num_list(list_):
    result = []
    for item in list_:
        if not isinstance(item, (int, float)):
            try:
                num = float(item)
            except (ValueError, TypeError):
                num = item
                if item=="NA":
                    num=None
        else:
            num = item
        result.append(num)
    return result


def has_object_type(ax_metadata, group, type_name):
    group_list = ax_metadata.get("object", {}).get(group, [])
    return any(obj.get("object_type") == type_name for obj in group_list)


def generate_grouped_or_stacked_addplot(zipped_group_data, layout, orientation, group_count=None, bar_shift=0.25):
    if layout == 'grouped' and (group_count is None or bar_shift is None):
        raise ValueError("group_count and bar_shift must be provided for grouped layout")

    lines = []

    for group_index, (group_data, color) in enumerate(zipped_group_data):
        if layout == 'grouped':
            shift = (group_index - (group_count - 1) / 2) * bar_shift
            shift_str = f", bar shift={shift:.3f}"
        else:
            shift_str = ""
        
        if orientation == 'vertical':
            bar_type = "ybar"
        elif orientation == 'horizontal':
            bar_type = "xbar"
        else:
            raise ValueError("orientation must be 'vertical' or 'horizontal'")

        lines.append(f"\\addplot+[ {bar_type}, fill={color}{shift_str} ] coordinates {{")

        for tick_index, value in enumerate(group_data):
            if orientation == 'vertical':
                coord = f"({tick_index}, {value})"
            else:
                coord = f"({value}, {tick_index})"

            lines.append(f"    {coord}")

        lines.append("};\n")

    return "\n".join(lines)


def convert_numeric_strings(lst):
    def try_convert(val):
        if isinstance(val, str) and val.replace('.', '', 1).isdigit():
            try:
                return float(val)
            except ValueError:
                return val
        return val

    if all(isinstance(el, (int, float)) for el in lst):
        return lst
    elif all(isinstance(el, str) and el.replace('.', '', 1).isdigit() for el in lst):
        return [float(el) for el in lst]
    elif all(isinstance(el, list) for el in lst):
        return [
            [float(e) if isinstance(e, str) and e.replace('.', '', 1).isdigit() else e for e in sub]
            for sub in lst
        ]
    else:
        return lst


def split_r_radarchart(code: str, num: int):
    lines = code.splitlines()

    import_packages = [l for l in lines if 'library(' in l]
    import_packages.append("library(ggplotify)")
    import_packages = '\n'.join(import_packages)

    code_lines = [l for l in lines if 'library(' not in l and 'dev.off()' not in l]
    code_lines = f"p{num} <- " + "as.ggplot(function() {\nop <- par(mar = c(1.5, 1.5, 2, 1)); on.exit(par(op), add = TRUE)\n" + '\n'.join(code_lines) + "\n})"

    return import_packages, code_lines


def split_tex_simple(tex: str, is_polar=False):
    lines = tex.splitlines()

    color_idx = [i for i, ln in enumerate(lines) if r'\definecolor' in ln]
    if not color_idx:
        before = tex
        colors = ""
        after = ""
        return before.strip(), colors, after

    first = color_idx[0]
    last  = color_idx[-1]

    before = "\n".join(lines[:first]).strip()

    colors = "\n".join(lines[i] for i in color_idx).strip()

    after_src = lines[last + 1 :]

    drop_if_contains = [
        r'\begin{document}',
        r'\end{document}',
        r'\begin{tikzpicture}',
        r'\end{tikzpicture}',
        r'\end{axis}',
        'width=',
        'height=',
    ]

    filtered = []
    for ln in after_src:
        if any(key in ln for key in drop_if_contains):
            continue
        ln = ln.replace(r'\begin{axis}', r'\nextgroupplot')
        filtered.append(ln)

    after_body = "\n".join(filtered).strip()

    if r'\pie' in after_body:
        after = (
            "\\nextgroupplot[\n"
            "  axis lines=none, ticks=none,\n"
            "  xmin=0, xmax=1, ymin=0, ymax=1,\n"
            "  clip=false\n"
            "]\n"
            "\\node[inner sep=0pt] at (rel axis cs:0.5,0.5) {\n"
            "  \\begin{tikzpicture}[baseline]\n"
            f"{after_body}\n"
            "  \\end{tikzpicture}\n"
            "};"
        ).strip()
    elif is_polar:
        after = (
            "\\begin{tikzpicture}\n"
            f"{after_body}\n"
            "\\end{tikzpicture}\n"
        ).strip()
    else:
        after = after_body

    return before, colors, after