import os
import json
import re
import statistics
import pandas as pd
import numpy as np
from copy import deepcopy
from collections import defaultdict, Counter

from worker.format_worker import *
from worker.value_mapping import *


def is_box_chart(ax_metadata, orig_lang):
    if orig_lang == "python":
        obj = ax_metadata.get("object", {})
        patch_count = sum(1 for p in obj.get("patches", []) if p.get("object_type") == "PathPatch")
        line_count = sum(1 for l in obj.get("lines", []) if l.get("object_type") == "Line2D")
        return patch_count >= 3 or (patch_count >= 1 and line_count >= 5)
    elif orig_lang == "r":
        patches = ax_metadata.get("object", {}).get("patches", [])
        return any(p.get("object_type") == "GeomBoxplot" for p in patches)
    return False


def infer_box_subtype(ax_metadata, orientation, orig_lang):
    patches = ax_metadata.get("object", {}).get("patches", [])
    lines = ax_metadata.get("object", {}).get("lines", [])
    collections = ax_metadata.get("object", {}).get("collections", [])
    if not patches:
        return "base-box"

    key_types = {"PathPatch", "GeomBoxplot",}
    box_objs = [
        p for p in patches
        if p.get("object_type") in key_types and "geometry" in p
    ]

    if not box_objs:
        return "base-box"
    
    is_grouped = False
    facecolors = [box.get("facecolor") for box in box_objs if box.get("facecolor") is not None]
    facecolor_counts = Counter(facecolors)
    num_unique_colors = len(facecolor_counts)
    is_grouped = any(count > 1 for count in facecolor_counts.values()) and num_unique_colors > 1

    has_dots = len(collections)==len(box_objs)

    miss_line = False
    is_notched = False
    geo_x = "x" if orientation=='vertical' else "y"
    if orig_lang=='python':
        if len(lines) > 0:
            has_line = any(
                len(set(line.get("geometry", {}).get(geo_x, []))) == 1 for line in lines
            )
            miss_line = has_line==None
        is_notched = any(patch.get("n_vertices", 6) > 6 for patch in box_objs)
    elif orig_lang=='r':
        miss_line = False
        is_notched = any(patch.get("geometry", {}).get("is_notched", None) != None for patch in box_objs)

    tags = []
    if is_grouped:
        tags.append("grouped")
    if miss_line:
        tags.append("missing")
    if has_dots:
        tags.append("dotted")
    if is_notched:
        tags.append("notched")
    return "-".join(tags) + "-box" if tags else "base-box"


def extract_box_stats_from_patch_and_lines(patch, lines, is_notched, geo_height, geo_width, geo_x, geo_y, x_tolerance=0.1):
    geom = patch["geometry"]
    q1 = float(geom[geo_y])
    q3 = q1 + float(geom[geo_height])
    xmin = float(geom[geo_x])
    xmax = xmin + float(geom[geo_width])
    xcenter = (xmin + xmax) / 2

    med = None
    whislo = None
    whishi = None

    for line in lines:
        x = [float(v) for v in line["geometry"][geo_x]]
        y = [float(v) for v in line["geometry"][geo_y]]
        if x==[] and y==[]:
            continue
        if len(x) < 2:
            continue
            continue
        if max(x) < xmin - x_tolerance or min(x) > xmax + x_tolerance:
            continue
        if (
            len(set(y)) == 1 and
            xmin - x_tolerance <= x[0] <= xmax + x_tolerance and 
            xmin - x_tolerance <= x[1] <= xmax + x_tolerance
        ):
            med = float(y[0])
        elif len(set(x)) == 1 and len(set(y)) == 2 and abs(x[0] - xcenter) <= x_tolerance:
            y_sorted = sorted(y)
            if y_sorted[0] < q1 or is_notched:
                whislo = y_sorted[0]
            if y_sorted[1] > q3 or is_notched:
                whishi = y_sorted[1]

    return {
        "q1": q1,
        "q3": q3,
        "med": med,
        "whislo": whislo,
        "whishi": whishi,
        "xmin": xmin,
        "xmax": xmax
    }



def extract_box_chart_data(metadata, ax_key, orig_lang, trans_lang, types, orientation, tick_labels, tick_pos, annotations, subtype):
    ax_metadata = metadata[ax_key]
    
    if orientation=='vertical' or orig_lang=='r':
        geo_height = "height"
        geo_width = "width"
        geo_x = "x"
        geo_y = "y"
    else:
        geo_height = "width"
        geo_width = "height"
        geo_x = "y"
        geo_y = "x"

    patches = [p for p in ax_metadata.get("object", {}).get("patches", []) if p.get("object_type") in {"PathPatch", "GeomBoxplot",}]
    lines = ax_metadata.get("object", {}).get("lines", [])
    collections = ax_metadata.get("object", {}).get("collections", [])
    is_notched = True if "notched" in subtype else False
    

    label_to_color = ax_metadata.get("type_agnostic", {}).get("label_to_color", {})
    if label_to_color=={} and 'grouped' in subtype:
        facecolors = []
        seen = set()
        for p in patches:
            c = p.get("facecolor")
            if c and c not in seen:
                facecolors.append(c)
                seen.add(c)
        label_to_color = {f"group {i+1}": c for i, c in enumerate(facecolors)}
    group_labels = list(label_to_color.keys()) if label_to_color else []

    sorted_patches = sorted(patches, key=lambda p: p["geometry"].get(geo_x, float("inf")))

    if orig_lang == 'r':
        q1     = [round(float(p["geometry"]["lower"]), 2)  for p in sorted_patches]
        med    = [round(float(p["geometry"]["middle"]), 2) for p in sorted_patches]
        q3     = [round(float(p["geometry"]["upper"]), 2)  for p in sorted_patches]
        whislo = [round(float(p["geometry"][geo_y]), 2) for p in sorted_patches]
        whishi = [round(float(p["geometry"][geo_y]) + float(p["geometry"]["height"]), 2) for p in sorted_patches]
        xmin   = [round(float(p["geometry"][geo_x]), 2) for p in sorted_patches]
        xmax   = [round(float(p["geometry"][geo_x]) + float(p["geometry"]["width"]), 2) for p in sorted_patches]
    else:
        q1, med, q3, whislo, whishi, xmin, xmax = [], [], [], [], [], [], []
        for p in sorted_patches:
            stat = extract_box_stats_from_patch_and_lines(p, lines, is_notched, geo_height, geo_width, geo_x, geo_y)
            q1.append(round(stat["q1"], 2))
            med.append(round(stat["med"], 2))
            q3.append(round(stat["q3"], 2))
            whislo.append(round(stat["whislo"], 2))
            whishi.append(round(stat["whishi"], 2))
            xmin.append(round(stat["xmin"], 2))
            xmax.append(round(stat["xmax"], 2))
    box_width = xmax[0] - xmin[0]
    box_alphas = sorted_patches[0]["alpha"]
    colors = [rgb_percent_to_hex(p["facecolor"]) for p in sorted_patches]
    if "grouped" in subtype:
        group = [find_label_by_color(color, label_to_color) for color in colors]
        group_stats = {g: {"q1": [], "med": [], "q3": [], "whislo": [], "whishi": [], "xmin": [], "xmax": [], "colors": []} for g in group_labels}
        for i, g in enumerate(group):
            group_stats[g]["q1"].append(q1[i])
            group_stats[g]["med"].append(med[i])
            group_stats[g]["q3"].append(q3[i])
            group_stats[g]["whislo"].append(whislo[i])
            group_stats[g]["whishi"].append(whishi[i])
            group_stats[g]["xmin"].append(xmin[i])
            group_stats[g]["xmax"].append(xmax[i])
            group_stats[g]["colors"].append(colors[i])
        
        q1     = [group_stats[g]["q1"]     for g in group_labels]
        med    = [group_stats[g]["med"]    for g in group_labels]
        q3     = [group_stats[g]["q3"]     for g in group_labels]
        whislo = [group_stats[g]["whislo"] for g in group_labels]
        whishi = [group_stats[g]["whishi"] for g in group_labels]
        xmin   = [group_stats[g]["xmin"]   for g in group_labels]
        xmax   = [group_stats[g]["xmax"]   for g in group_labels]
        colors = [group_stats[g]["colors"][0] for g in group_labels]
    
    x_dot = []
    y_dot = []
    dot_color = []
    dot_shape = []
    dot_size = []
    dot_alpha = []
    has_dotted = False
    if "dotted" in subtype:
        has_dotted = True
        for col in collections:
            facecolors = col.get("facecolors", [None])
            sizes = col.get("sizes", [None])
            shapes = col.get("shape", [None])
            alphas = col.get("alpha", [None])
            facecolors = facecolors if isinstance(facecolors, (list, tuple)) else [facecolors]
            shapes     = shapes     if isinstance(shapes, (list, tuple))     else [shapes]
            sizes      = sizes      if isinstance(sizes, (list, tuple))      else [sizes]
            alphas     = alphas     if isinstance(alphas, (list, tuple))     else [alphas]

            x_dot.append([pos[0] for pos in col["geometry"]])
            y_dot.append([pos[1] for pos in col["geometry"]])
            dot_color.append(rgb_percent_to_hex(facecolors[0]))
            dot_shape.append(map_marker_style(shapes[0], orig_lang, trans_lang))
            dot_size.append(map_marker_size(sizes[0], orig_lang, trans_lang))
            dot_alpha.append(alphas[0])


    template = "box"
    tags = []
    template += f"_{orientation}"
    if "grouped" in subtype:
        template += '_grouped'
    if "dotted" in subtype:
        template += '_dotted'
    template += f'_{trans_lang}.jinja'

    background_color = rgb_percent_to_hex(ax_metadata["type_agnostic"].get("background_color") or '#Ffffff')
    dot_alpha = dot_alpha[0] if dot_alpha!=[] else dot_alpha
    if trans_lang=='latex':
        all_colors = [colors, dot_color]
        color_define_str, color_labels = generate_latex_color_define(all_colors, background_color, ax_key)
        colors = color_labels[0]
        dot_color = color_labels[1]
        background_color = 'cb'

        group_labels = [escape_string(item, trans_lang) for item in group_labels]
        if "grouped" not in subtype:
            box_pos =[i+1 for i in range(len(tick_labels))]
        else:
            box_pos = [len(xmin) * (j+1) for j in range(len(xmin[0]))]
            print(box_pos)
        box_pos = format_list_for_lang(box_pos, trans_lang, "num")
        tick_labels_str = format_list_for_lang(tick_labels, trans_lang, "str")
    else:
        color_define_str = ""
        tick_labels_str = ""
        box_pos = []
        group_labels = format_list_for_lang(group_labels, trans_lang, "str")
        tick_labels = format_list_for_lang(tick_labels, trans_lang, "str")
        colors = format_list_for_lang(colors, trans_lang, "str")
        q1 = format_list_for_lang(q1, trans_lang, "num")
        med = format_list_for_lang(med, trans_lang, "num")
        q3 = format_list_for_lang(q3, trans_lang, "num")
        whislo = format_list_for_lang(whislo, trans_lang, "num")
        whishi = format_list_for_lang(whishi, trans_lang, "num")
        xmin = format_list_for_lang(xmin, trans_lang, "num")
        xmax = format_list_for_lang(xmax, trans_lang, "num")
        x_dot = format_list_for_lang(x_dot, trans_lang, "num")
        y_dot = format_list_for_lang(y_dot, trans_lang, "num")
        dot_color = format_list_for_lang(dot_color, trans_lang, "str")
        dot_shape = format_list_for_lang(dot_shape, trans_lang, "str")
        dot_size = format_list_for_lang(dot_size, trans_lang, "num")

    context = {
        "group_labels": group_labels,
        "tick_labels": tick_labels,
        "colors": colors,
        "alphas": box_alphas,
        "q1": q1,
        "med": med,
        "q3": q3,
        "whislo": whislo,
        "whishi": whishi,
        "xmin": xmin,
        "xmax": xmax,
        "box_width": box_width,

        "has_dotted": has_dotted,
        "x_dot": x_dot,
        "y_dot": y_dot,
        "dot_color": dot_color,
        "dot_shape": dot_shape,
        "dot_size": dot_size,
        "dot_alpha": dot_alpha,
        
        "is_notched": is_notched,

        "color_define": color_define_str,
        "background_color": background_color,
        "box_pos": box_pos,
        "tick_labels_str": tick_labels_str,

        "annotations": annotations,
        "template_file": template
    }
    return context