import os
import json
import pandas as pd
from copy import deepcopy
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import matplotlib.collections as mcollections

import argparse
from pathlib import Path

import re
from io import StringIO
import random
import jsonlines
import shortuuid
from openai import OpenAI
import regex
import copy

from type_classifier import infer_orientation, infer_types, infer_subtypes 
from format_worker import rgb_percent_to_hex

def default_metadata_template(num_ax):
    metadata = {
        "execute": {
            "source_file": None,
            "language": None,
            "executable": None
        },
        "suptitle": None,
        "legend": None,
        "plot_size": {"width": 6.4, "height": 4.8, "unit": "inch"},
        "twin_axes": {},
        "axes_layout": {},
        "facecolor": None,
    }
    for k in range(num_ax):
        metadata[f'ax_{k}'] = default_metadata_ax()
    return metadata


def default_metadata_ax():
    return {
        "type_agnostic": {
            "axis": {},
            "title": {},
            "x_label": {},
            "y_label": {},
            "x_ticks": [],
            "y_ticks": [],
            "legend": {},
            "grid": {},
            "panel_box": False,
            "background_color": None,
            'annotation': [],
            "label_to_color": {},
            "container_type": [],
        },
        "type_specific": {
            "type": None,
            "sub_type": None,
            "orientation": None,
        },
        "object": {
            "patches": [],
            "lines": [],
            "collections": []
        }
    }

def remove_latex_comments(code_str):
    cleaned_lines = []
    for line in code_str.splitlines():
        line_no_comment = re.split(r'(?<!\\)%', line)[0]
        cleaned_lines.append(line_no_comment.rstrip())
    return '\n'.join(cleaned_lines)

def strip_json_comments(json_str):
    cleaned_lines = []
    for line in json_str.splitlines():
        cleaned_line = re.sub(r'//.*$', '', line)
        if cleaned_line.strip():
            cleaned_lines.append(cleaned_line.strip())
    return '\n'.join(cleaned_lines)

def detect_latex_subplots(code_str):
    matches = list(re.finditer(r'\\begin\s*{axis}(.*?)\\end\s*{axis}', code_str, re.DOTALL))
    subplot_blocks = [f'\\begin{{axis}}{m.group(1)}\\end{{axis}}' for m in matches]
    num_subplots = len(subplot_blocks)
    has_axis = num_subplots > 0
    group_size_match = re.search(
        r'group\s*style\s*=\s*{[^}]*group\s+size\s*=\s*(\d+)\s*by\s*(\d+)', code_str
    )
    if group_size_match:
        n_cols = int(group_size_match.group(1))
        n_rows = int(group_size_match.group(2))
    else:
        n_cols = None
        n_rows = None
    twin_axis_indicators = [
        "y axis on right",
        "x axis top",
        "axis y line* = right",
        "axis x line* = top",
        "x label style = right",
        "y label style = top",
        "enlargelimits = false",
        "axis line style =",
        "tick align = outside",
        "axis background/.style",
        "hide x axis",
        "hide y axis",
        "x tick label style =",
        "y tick label style =",
    ]
    has_twin_axes = False
    for block in subplot_blocks:
        match = re.search(r'\\begin\s*{axis}(\[.*?\])?', block, re.DOTALL)
        if match and match.group(1):
            lower = match.group(1).lower()
            if any(ind in lower for ind in twin_axis_indicators):
                has_twin_axes = True
                break

    return has_axis, num_subplots, subplot_blocks, n_rows, n_cols, has_twin_axes

def detect_twin_axes_latex(code_str):
    axis_blocks = re.findall(r'\\begin\s*{axis}(\[.*?\])?', code_str, re.DOTALL)
    twin_candidates = []
    for block in axis_blocks:
        if block:
            lower = block.lower()
            if ("y axis on right" in lower or
                "x axis top" in lower or
                "at=" in lower or
                "anchor=" in lower):
                twin_candidates.append(block)

    twin_detected = len(twin_candidates) >= 1 and len(axis_blocks) >= 2

    return twin_detected, twin_candidates

def extract_axis_options(code_str):
    match = regex.search(r'\\begin{axis}\s*\[((?:[^\[\]]++|(?R))*?)\]', code_str, regex.DOTALL)
    return match.group(1).strip() if match else None


def extract_ticks_from_latex_code(code_str):
    axis_match = re.search(r'\\begin\s*{axis}\s*\[([^\]]+)\]', code_str, re.DOTALL)
    if not axis_match:
        return [], []
    options = axis_match.group(1)
    x_ticks, y_ticks = [], []
    symbolic_x_match = re.search(r'symbolic\s+x\s+coords\s*=\s*{([^}]+)}', options, re.IGNORECASE)
    if symbolic_x_match:
        symbols = [s.strip() for s in symbolic_x_match.group(1).split(',')]
        x_ticks = [{"text": s, "position": [i, 0]} for i, s in enumerate(symbols)]
    symbolic_y_match = re.search(r'symbolic\s+y\s+coords\s*=\s*{([^}]+)}', options, re.IGNORECASE)
    if symbolic_y_match:
        symbols = [s.strip() for s in symbolic_y_match.group(1).split(',')]
        y_ticks = [{"text": s, "position": [0, i]} for i, s in enumerate(symbols)]
    xtick_match = re.search(r'xtick\s*=\s*{([^}]+)}', options, re.IGNORECASE)
    if xtick_match:
        raw = xtick_match.group(1).strip()
        if raw != 'data' and not x_ticks:
            x_ticks = [{"text": t.strip(), "position": None} for t in raw.split(',')]
    ytick_match = re.search(r'ytick\s*=\s*{([^}]+)}', options, re.IGNORECASE)
    if ytick_match:
        raw = ytick_match.group(1).strip()
        if raw != 'data' and not y_ticks:
            y_ticks = [{"text": t.strip(), "position": None} for t in raw.split(',')]

    return x_ticks, y_ticks


def extract_font_style(style_block):
    size_map = ["\\tiny", "\\scriptsize", "\\footnotesize", "\\small", "\\normalsize", "\\large", "\\Large", "\\LARGE", "\\huge", "\\Huge"]
    style = {"size": None, "style": "normal"}
    font_match = re.search(r'font\s*=\s*([^\],]+)', style_block)
    if font_match:
        font_str = font_match.group(1)
        for size in size_map:
            if size in font_str:
                style["size"] = size
                break
        if "\\bfseries" in font_str:
            style["style"] = "bold"
        elif "\\itshape" in font_str or "\\italic" in font_str:
            style["style"] = "italic"
    return style

def extract_legend_location(options):
    pos_match = re.search(r'legend\s+pos\s*=\s*([a-zA-Z ]+)', options)
    if pos_match:
        return pos_match.group(1).strip()
    style_match = regex.search(r'legend\s+style\s*=\s*({(?:[^{}]|(?1))*})', options, regex.DOTALL)
    if style_match:
        style_content = style_match.group(1)

        at_match = re.search(r'at\s*=\s*\{?\(?\s*([0-9.\-]+)\s*,\s*([0-9.\-]+)\s*\)?\}?', style_content)
        anchor_match = re.search(r'anchor\s*=\s*([a-zA-Z ]+)', style_content)

        if at_match and anchor_match:
            x, y = float(at_match.group(1)), float(at_match.group(2))
            anchor = anchor_match.group(1).strip().lower()
            mapping = {
                ((1.0, 1.0), "north east"): "north east",
                ((0.0, 1.0), "north west"): "north west",
                ((1.0, 0.0), "south east"): "south east",
                ((0.0, 0.0), "south west"): "south west",
                ((0.5, 1.0), "north"): "north",
                ((0.5, 0.0), "south"): "south",
                ((1.0, 0.5), "east"): "east",
                ((0.0, 0.5), "west"): "west",
                ((0.5, 0.5), "center"): "center",
                ((0.5, 1.1), "south"): "outer north",
                ((0.5, -0.1), "north"): "outer south",
                ((1.1, 0.5), "west"): "outer east",
                ((-0.1, 0.5), "east"): "outer west",
                ((1.1, 1.1), "south west"): "outer north east",
                ((-0.1, 1.1), "south east"): "outer north west"
            }
            for (px, py), anch in mapping:
                if abs(x - px) < 0.05 and abs(y - py) < 0.05 and anchor == anch:
                    return mapping[(px, py), anch]
    return "outer north east"

def normalize_label(raw_label):
    if not raw_label:
        return None
    label = raw_label.strip()
    if label.startswith("$") and label.endswith("$"):
        inner = label[1:-1].strip()
        return f"${inner.replace('\\', '\\\\')}$"
    else:
        return label

def parse_style_content(style_str):
    d = {}
    for s in style_str.split(","):
        s = s.strip()
        if "=" in s:
            k, v = s.split("=", 1)
            d[k.strip()] = v.strip()
        else:
            d[s] = True
    return d

def parse_coordinates(coord_str):
    if not coord_str:
        return None
    if ":" in coord_str or coord_str.startswith("+"):
        return None
    try:
        parts = [float(x.strip().replace("cm", "")) for x in coord_str.split(",")]
        if len(parts) == 2:
            return parts
    except ValueError:
        pass
    return None

def extract_tikz_styles(code_str):

    style_metadata = {}
    tikzstyle_pattern = re.compile(r'\\tikzstyle\s*{(\w+)}\s*=\s*\[([\s\S]*?)\]', re.DOTALL)
    for match in tikzstyle_pattern.finditer(code_str):
        style_name = match.group(1).strip()
        style_attrs = match.group(2).replace("\n", " ").strip()
        style_metadata[style_name] = parse_style_content(style_attrs)
    tikzpicture_style_pattern = re.compile(r'\\begin\s*{tikzpicture}\s*\[\s*([^\]]+)\s*\]', re.DOTALL)
    for match in tikzpicture_style_pattern.finditer(code_str):
        style_block = match.group(1).strip()
        for attr in re.split(r',(?![^{]*\})', style_block):
            attr = attr.strip()
            if not attr:
                continue
            if '.style=' in attr:
                name, value = attr.split('.style=', 1)
                style_metadata[name.strip()] = parse_style_content(value.strip('{}'))
            elif '.style args=' in attr:
                name, value = attr.split('.style args=', 1)
                style_metadata[name.strip()] = {"raw_args": value.strip()}
            elif '=' in attr:
                key, val = attr.split('=', 1)
                style_metadata[key.strip()] = val.strip()
            else:
                style_metadata[attr] = True

    return style_metadata

def extract_node_in_graph(code_str):

    obj_metadata = []

    coord_pattern = re.compile(r'\\coordinate\s*\(([\w+-]+)\)\s*at\s*\(([^)]+)\)\s*;', re.DOTALL)
    for match in coord_pattern.finditer(code_str):
        node_id, coord_str = match.groups()
        coords = parse_coordinates(coord_str)
        obj_metadata.append({
            "id": node_id,
            "label": None,
            "style": None,
            "position": {
                "type": "absolute",
                "x": coords[0] if coords else None,
                "y": coords[1] if coords else None
            },
            "label_position": None
        })
    
    node_pattern = re.compile(
        r'\\node\s*(\[[^\]]*\])?\s*\(([^)]+)\)\s*(?:at\s*\(([^)]+)\))?\s*{(.*?)}\s*;',
        re.DOTALL
    )

    for match in node_pattern.finditer(code_str):
        raw_style, node_id, coord_str, raw_label = match.groups()
        label = normalize_label(raw_label.strip()) if raw_label else None
        label_position = None
        parsed_style = None

        if raw_style:
            style_parts = parse_style_content(raw_style.strip("[]"))

            filtered_parts = []
            positional_prefixes = ("right of", "left of", "above of", "below of", "node distance", "on grid")
            for part in style_parts:
                if part.startswith("vrtx=") and "/" in part:
                    pos, val = part[len("vrtx="):].split("/", 1)
                    label = normalize_label(val)
                    label_position = pos.strip()
                elif any(part.startswith(prefix) for prefix in positional_prefixes):
                    continue
                else:
                    filtered_parts.append(part)

            parsed_style = ", ".join(filtered_parts) if filtered_parts else None

        coords = parse_coordinates(coord_str)
        obj_metadata.append({
            "id": node_id,
            "label": label,
            "style": parsed_style,
            "position": {
                "type": "absolute",
                "x": coords[0] if coords else None,
                "y": coords[1] if coords else None
            },
            "label_position": label_position
        })

    return obj_metadata


def extract_shape_in_graph(code_str):

    obj_metadata = []

    draw_pattern = re.compile(
        r"\\(draw|fill|filldraw)\s*(\[[^\]]*\])?\s*(.*?)\s*;",
        re.DOTALL
    )

    for _, match in enumerate(draw_pattern.finditer(code_str), start=1):
        cmd_type, style_block, path_content = match.groups()
        path_content = path_content.strip()
        if (
            not style_block
            and not re.search(r"\d+\s*,\s*\d+", path_content)
            and re.match(r"\(.+?\)\s+edge\s+\(.+?\)", path_content)
        ):
            continue
        style_dict = {}
        if style_block:
            for attr in style_block.strip("[]").split(","):
                attr = attr.strip()
                if "=" in attr:
                    k, v = attr.split("=", 1)
                    style_dict[k.strip()] = v.strip()
                else:
                    style_dict[attr] = True
                    if attr in {
                        "ultra thick", "very thick", "thick",
                        "semithick", "thin", "very thin"
                    }:
                        style_dict["line width"] = attr
        
        if "rectangle" in path_content:
            shape_type = "rectangle"
        elif "circle" in path_content:
            shape_type = "circle"
        elif "grid" in path_content:
            shape_type = "grid"
        elif "controls" in path_content:
            shape_type = "bezier_path"
        else:
            shape_type = "path"

        shape_entry = {
            "type": shape_type,
            "style": None,
            "fill": style_dict.get("fill"),
            "draw": style_dict.get("draw"),
            "line_width": style_dict.get("line width"),
            "color": style_dict.get("color") or None,
            "position": None,
            "segments": None
        }

        if shape_type == "rectangle":
            m = re.search(r"\(([^)]+)\)\s+rectangle\s+\(([^)]+)\)", path_content)
            if m:
                try:
                    p1 = [float(x) for x in m.group(1).split(",")]
                    p2 = [float(x) for x in m.group(2).split(",")]
                    shape_entry["position"] = [p1, p2]
                except ValueError:
                    pass
        
        elif shape_type == "circle":
            m = re.search(r"\(([^)]+)\)\s+circle\s+\(([^)]+)\)", path_content)
            if m:
                center_str, radius_str = m.group(1).strip(), m.group(2).strip()
                try:
                    radius = float(re.sub(r"[a-zA-Z]+", "", radius_str))
                except ValueError:
                    radius = None

                if ":" in center_str:
                    try:
                        angle, length = center_str.split(":")
                        shape_entry["position"] = {
                            "polar": True,
                            "angle_deg": float(angle.strip()),
                            "length": float(re.sub(r"[a-zA-Z]+", "", length.strip())),
                            "radius": radius
                        }
                    except ValueError:
                        shape_entry["position"] = None
                else:
                    try:
                        center = [float(x.strip().replace("cm", "")) for x in center_str.split(",")]
                        shape_entry["position"] = {
                            "center": center,
                            "radius": radius
                        }
                    except ValueError:
                        shape_entry["position"] = None
        
        if shape_type in {"path", "bezier_path"}:
            point_pattern = re.findall(r"\(([^)]+)\)", path_content)
            segments = []
            for pt in point_pattern:
                coords = [x.strip() for x in pt.split(",")]
                if len(coords) == 2:
                    try:
                        segments.append([float(c) for c in coords])
                    except ValueError:
                        continue
            if segments:
                shape_entry["segments"] = segments
                if "cycle" in path_content:
                    shape_entry["closed"] = True
        
        if (
            shape_type in {"rectangle", "circle", "grid"}
            or shape_entry.get("segments")
            or shape_entry.get("fill")
            or shape_entry.get("draw")
        ):
            obj_metadata.append(shape_entry)

    return obj_metadata


def extract_edge_in_graph(code_str):

    obj_metadata = []
    edge_pattern = re.compile(
        r"\\(?:draw|path)\s*(\[[^\]]*\])?\s*"
        r"\(([^)]+)\)\s*edge(?:\s*node\s*\[[^\]]*\]\s*{([^}]*)})?\s*"
        r"\(([^)]+)\)\s*;",
        re.DOTALL
    )
    for match in edge_pattern.finditer(code_str):
        style_block, from_node, edge_label, to_node = match.groups()
        edge = {
            "from": from_node.strip(),
            "to": to_node.strip(),
            "label": normalize_label(edge_label) if edge_label else None,
            "style": style_block.strip("[]") if style_block else None,
            "label_position": "midway" if edge_label else None
        }
        obj_metadata.append(edge)
    
    draw_line_pattern = re.compile(
        r"\\draw\s*(\[[^\]]*\])?\s*\(([^)]+)\)\s*--\s*\(([^)]+)\)\s*;",
        re.DOTALL
    )
    for match in draw_line_pattern.finditer(code_str):
        style_block, from_node, to_node = match.groups()
        edge = {
            "from": from_node.strip(),
            "to": to_node.strip(),
            "label": None,
            "style": style_block.strip("[]") if style_block else None,
            "label_position": None
        }
        obj_metadata.append(edge)

    return obj_metadata

def extract_metadata_graph(code_str, label="script.tex"):
    metadata = {
        "execute": {
            "source_file": label,
            "language": "latex",
            "executable": True
        },
        "plot_size": {"width": 6.4, "height": 4.8, "unit": "inch"},
        "facecolor": None,
        "style": {},
        "object": {
            "shape": [],
            "node": [],
            "edge": []
        }
    }
    metadata["style"] = extract_tikz_styles(code_str)
    metadata["object"]["node"] = extract_node_in_graph(code_str)
    metadata["object"]["shape"] = extract_shape_in_graph(code_str)
    metadata["object"]["edge"] = extract_edge_in_graph(code_str)
    return metadata

def extract_metadata_from_latex_runtime(code_str, label="script.tex"):
    code_str = remove_latex_comments(code_str)
    has_axis, num_subplots, subplot_blocks, n_rows, n_cols, has_twin_axes = detect_latex_subplots(code_str)
    if not has_axis:
        print("Not chart type -> Graph: {}".format(label))
        metadata = extract_metadata_graph(code_str, label)
        return metadata
    
    metadata = default_metadata_template(num_subplots)
    metadata["execute"] = {
        "language": "latex", "source_file": label, "executable": True
    }
    metadata["suptitle"] = None
    metadata["legend"] = {
        'exist': False, 'loc': None, 'ncol': None, 'labels': None
    }
    metadata["facecolor"] = None
    metadata["twin_axes"] = has_twin_axes
    metadata["axes_layout"] = {"n_row": n_rows, "n_col": n_cols}

    if num_subplots==0:
        return metadata
    
    for k in range(num_subplots):
        ax_str = subplot_blocks[k]

        options = extract_axis_options(ax_str)
        if not options:
            options = copy.copy(ax_str)

        def extract_option(key):
            match = re.search(rf'{key}\s*=\s*([^\n,]+)', options)
            return match.group(1).strip() if match else None

        for dim in ["width", "height"]:
            val = extract_option(dim)
            if val: 
                unit = "cm" if val.endswith("cm") else "inch" if val.endswith("in") else None
                if unit:
                    metadata['plot_size']["unit"] = unit
                    metadata["plot_size"][dim] = round(float(val.replace(unit, "").strip()), 2)
                else:
                    metadata["plot_size"][dim] = val

        metadata[f'ax_{k}']["type_agnostic"]["ax_str"] = ax_str
        metadata[f'ax_{k}']["type_agnostic"]["axis"] = {
            "position": None, "type": None, "aspect":None
        }

        label_style = re.search(r'label\s*style\s*=\s*{([^}]*)}', options)
        title_style = re.search(r'title\s*style\s*=\s*{([^}]*)}', options)
        label_font = extract_font_style(label_style.group(1)) if label_style else {"size": "\\normalsize", "style": "normal"}
        title_font = extract_font_style(title_style.group(1)) if title_style else {"size": "\\normalsize", "style": "normal"}

        metadata[f'ax_{k}']["type_agnostic"]["title"] = {
            'content': extract_option("title"), 
            'size': title_font["size"], 
            'style': title_font["style"]
        }

        metadata[f'ax_{k}']["type_agnostic"]["x_label"] = {
            'content': extract_option("xlabel"), 
            'size': label_font["size"], 
            'style': label_font["style"]
        }

        metadata[f'ax_{k}']["type_agnostic"]["y_label"] = {
            'content':  extract_option("ylabel"), 
            'size': label_font["size"], 
            'style':  label_font["style"]
        }

        x_ticks, y_ticks = extract_ticks_from_latex_code(ax_str)
        metadata[f'ax_{k}']["type_agnostic"]["x_ticks"] = x_ticks
        metadata[f'ax_{k}']["type_agnostic"]["y_ticks"] = y_ticks

        if 'legend style' in options or "\\legend" in ax_str:
            legend_style_match = re.search(r'legend\s+columns\s*=\s*(\d+)', options)
            metadata[f'ax_{k}']["type_agnostic"]["legend"] = {
                'exist': True, 
                'loc': extract_legend_location(options), 
                'ncol': int(legend_style_match.group(1)) if legend_style_match else None, 
            }
        else:
            metadata[f'ax_{k}']["type_agnostic"]["legend"] = {
                'exist': False, 'loc': None, 'ncol': None
            }

        axis_lines = extract_option("axis lines")
        if axis_lines or axis_lines=='box':
            metadata[f'ax_{k}']["type_agnostic"]["panel_box"] = True
        else:
            metadata[f'ax_{k}']["type_agnostic"]["panel_box"] = False

        options_lower = options.lower()
        metadata[f'ax_{k}']["type_agnostic"]["grid"] = {
            'x': (
                "xmajorgrids" in options_lower or 
                "xminorgrids" in options_lower or 
                "x grid style" in options_lower or 
                "grid=major" in options_lower
            ),
            'y': (
                "ymajorgrids" in options_lower or 
                "yminorgrids" in options_lower or 
                "y grid style" in options_lower or 
                "grid=major" in options_lower
            )
        }

        if "nodes near coords" in ax_str.lower():
            metadata[f'ax_{k}']["type_agnostic"]["annotation"] = [True]
        metadata[f'ax_{k}']["type_agnostic"]["background_color"] = None
        metadata[f'ax_{k}']["type_agnostic"]["container_type"] = []

        metadata[f'ax_{k}']["object"] = {}
        metadata[f'ax_{k}']["type_agnostic"]["label_to_color"] = {}

        metadata[f'ax_{k}']["type_specific"] = {
            "orientation": None,
            "type": None,
            "sub_type": None,
        }

    return metadata

if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Extract plot metadata from Python scripts.")
    p.add_argument("--head_dir")
    p.add_argument("--suffix")
    args = p.parse_args()

    head_dir = args.head_dir
    scripts = [f for f in os.listdir(head_dir) if f.endswith(args.suffix)]

    for script_pt in scripts:
        script_path = os.path.join(head_dir, script_pt)
        with open(script_path, "r") as f:
            code_str = f.read()
        metadata = extract_metadata_from_latex_runtime(code_str, label=script_path)
        save_path = script_path.split('.')[0] + '_object.json'
        json_string = json.dumps(metadata, indent=4, default=str)
        with open(save_path, 'w') as fp:
            fp.write(json_string)