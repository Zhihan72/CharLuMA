import os
import json
import re
import statistics
import math
import pandas as pd
import numpy as np
from copy import deepcopy
from collections import defaultdict, Counter

from worker.format_worker import *
from worker.value_mapping import *

def is_bar_chart(ax_metadata, orig_lang):
    if orig_lang == "python":
        axis_type = ax_metadata.get("type_agnostic", {}).get("axis", {}).get("type", "")
        if axis_type == "polar":
            return False
        container_types = ax_metadata.get("type_agnostic", {}).get("container_type", [])
        if "BarContainer" in container_types:
            return True
        patches = ax_metadata.get("object", {}).get("patches", [])
        return any(p.get("object_type") == "Rectangle" for p in patches)
    elif orig_lang == "r":
        axis_type = ax_metadata.get("type_agnostic", {}).get("axis", {}).get("type", "")
        if axis_type == "polar":
            return False
        patches = ax_metadata.get("object", {}).get("patches", [])
        return any(p.get("object_type") in ["GeomBar", "GeomCol", "GeomHistogram"] for p in patches)
    return False


def infer_bar_subtype(ax_metadata, orientation, orig_lang=None):
    if orig_lang == "r":
        patches = ax_metadata.get("object", {}).get("patches", [])
        if any(p.get("object_type") == "GeomHistogram" for p in patches):
            return "histogram"
    bars = []
    for p in ax_metadata.get("object", {}).get("patches", []):
        if p.get("object_type") in {"Rectangle", "GeomBar", "GeomCol", "GeomRect"} and "geometry" in p:
            g = p["geometry"]
            try:
                if orientation == "vertical":
                    s = float(g.get("x"))
                    e = float(g.get("x", 0.0)) + float(g.get("width", 0.0))
                else:
                    s = float(g.get("y"))
                    e = float(g.get("y", 0.0)) + float(g.get("height", 0.0))
            except Exception:
                continue
            if s is None or e is None:
                continue
            if s > e:
                s, e = e, s
            bars.append((s, e))

    if not bars:
        return "base-bar"

    bars.sort(key=lambda t: (t[0], t[1]))
    widths  = [e - s for s, e in bars]
    centers = [(s + e) * 0.5 for s, e in bars]
    span = max(bars[-1][1] - bars[0][0], 1e-9)
    tol  = max(1e-6, 0.002 * span)

    if any(w <= tol for w in widths):
        return "base-bar"
    
    same_interval_counts = {}
    for s, e in bars:
        key = (round(s / tol), round(e / tol)) if tol > 0 else (s, e)
        same_interval_counts[key] = same_interval_counts.get(key, 0) + 1
    if any(c >= 2 for c in same_interval_counts.values()):
        return "stacked-bar"
    
    gaps = [bars[i+1][0] - bars[i][1] for i in range(len(bars) - 1)]
    if not gaps:
        return "base-bar"

    med_w = sorted(widths)[len(widths) // 2]
    large_gap_thr = max(0.6 * med_w, 3 * tol)
    small_gap_thr = 0.5 * med_w + tol

    clusters, cur = [], [bars[0]]
    for i, g in enumerate(gaps):
        if g >= large_gap_thr:
            clusters.append(cur)
            cur = [bars[i+1]]
        else:
            cur.append(bars[i+1])
    clusters.append(cur)

    cluster_sizes = [len(c) for c in clusters]
    if len(clusters) >= 2 and max(cluster_sizes) > 1 and max(cluster_sizes) == min(cluster_sizes):
        rel_sets = []
        for c in clusters:
            c_centers = [(s + e) * 0.5 for s, e in c]
            c0 = sum(c_centers) / len(c_centers)
            rel = tuple(round(ci - c0, 6) for ci in c_centers)
            rel_sets.append(rel)
        if len(set(rel_sets)) == 1:
            return "grouped-bar"
    
    m = sum(widths) / len(widths)
    cv = (math.sqrt(sum((w - m) ** 2 for w in widths) / len(widths)) / abs(m)) if m else 0.0
    uniform_widths = cv <= 0.2
    touching_ratio = sum(abs(g) <= tol for g in gaps) / max(1, len(gaps))
    monotonic_centers = all(centers[i] < centers[i+1] - 1e-12 for i in range(len(centers) - 1))
    many_bins = len(bars) >= 5

    has_small = any((g > tol) and (g < small_gap_thr) for g in gaps)
    has_large = any(g >= large_gap_thr for g in gaps)
    unstable_clusters = len(set(cluster_sizes)) > 1 or max(cluster_sizes) <= 1

    if monotonic_centers and uniform_widths and (touching_ratio >= 0.7 or (has_small and has_large and unstable_clusters)) and many_bins:
        return "histogram"
    return "base-bar"


def extract_bar_chart_data(metadata, ax_key, orig_lang, trans_lang, types, orientation, tick_labels, tick_pos, annotations, subtype):
    ax_metadata = metadata[ax_key]
    layout = subtype.split('-')[0].strip()
    axis_type = ax_metadata.get("type_agnostic", {}).get("axis", {}).get("type", "")
    if orientation=='vertical' or orig_lang=='r':
        geo_val_pos = "height"
        geo_width_pos = "width"
        geo_tick_pos = "x"
    else:
        geo_val_pos = "width"
        geo_width_pos = "height"
        geo_tick_pos = "y"
    if orig_lang=='r' and orientation=='horizontal' and axis_type!= "flip":
        geo_val_pos = "width"
        geo_width_pos = "height"
        geo_tick_pos = "y"

    key_types = {"Rectangle", "GeomBar", "GeomCol", "GeomRect"}
    patches = [
        p for p in ax_metadata.get("object", {}).get("patches", [])
        if p.get("object_type") in key_types and "geometry" in p
    ]

    label_to_color = ax_metadata.get("type_agnostic", {}).get("label_to_color", [])
    group_labels = list(label_to_color.keys()) if label_to_color else []
    if subtype in {"grouped-bar", "stacked-bar"} and group_labels==[]:
        patch_colors_all = [rgb_percent_to_hex(p["facecolor"]) for p in patches]
        patch_colors_all = list(set(patch_colors_all))
        label_to_color = {f"group_{k+1}": patch_colors_all[k] for k in range(len(patch_colors_all))}
    
    if subtype in {"grouped-bar", "stacked-bar"}:
        values_dict = defaultdict(lambda: [0] * len(tick_labels))
        colors_dict = defaultdict(lambda: [None] * len(tick_labels))
        hatch_dict = defaultdict(lambda: [None] * len(tick_labels))
        tick_idx_map = {tick: i for i, tick in enumerate(tick_labels)}

        for k in range(len(patches)):
            patch = patches[k]
            patch_color = rgb_percent_to_hex(patch["facecolor"])
            geom = patch["geometry"]
            value = geom[geo_val_pos]
            pos = geom[geo_tick_pos]
            tick = find_nearest_tick_label(pos, tick_labels, tick_pos)
            group = find_label_by_color(patch_color, label_to_color)
            idx = tick_idx_map[tick]
            values_dict[group][idx] = float(value)
            colors_dict[group][idx] = patch_color
            hatch_dict[group][idx] = patch["hatch"]

        group_labels = list(values_dict.keys())
        values = [values_dict[g] for g in group_labels]
        colors = [colors_dict[g] for g in group_labels]
        hatches = [hatch_dict[g] for g in group_labels]
    else:
        sorted_patches = sorted(patches, key=lambda p: p["geometry"].get(geo_tick_pos, float("inf")))
        values = [float(p["geometry"][geo_val_pos]) for p in sorted_patches]
        colors = [rgb_percent_to_hex(p["facecolor"]) for p in sorted_patches]
        hatches = [p["hatch"] for p in sorted_patches]
    
    if subtype=="stacked-bar" and (orig_lang == "r" or trans_lang == "r"):
        group_labels, values, colors, hatches, annotations = (
            group_labels[::-1], values[::-1], colors[::-1], hatches[::-1], annotations[::-1]
        )
    
    background_color = rgb_percent_to_hex(ax_metadata["type_agnostic"].get("background_color") or '#Ffffff')
    if trans_lang=='latex':
        bar_pos =[i for i in range(len(tick_labels))]
        
        color_define_str, color_labels = generate_latex_color_define(colors, background_color, ax_key)
        background_color = 'cb'
        color_labels = [item[0] if isinstance(item, list) else item for item in color_labels]
        colors = color_labels

        add_plots = ""
        if subtype in {"grouped-bar", "stacked-bar"}:
            zipped_group_data = list(zip(values, color_labels))
            add_plots = generate_grouped_or_stacked_addplot(zipped_group_data, layout, orientation, len(group_labels), bar_width)
        bar_pos = format_list_for_lang(bar_pos, trans_lang, "num")
        group_labels = ["{"+escape_string(item, trans_lang)+"}" for item in group_labels if item!=None]
    else:
        bar_pos = []
        add_plots = ""
        color_define_str = ""
        group_labels = format_list_for_lang(group_labels, trans_lang, "str")
        values = format_list_for_lang(values, trans_lang, "num")
        colors = format_list_for_lang(colors, trans_lang, "str")
        hatches = format_list_for_lang(hatches, trans_lang, "str")
        annotations = format_list_for_lang(annotations, trans_lang, "str")
    
    context = {
        "group_labels": group_labels,
        "values": values,
        "colors": colors,
        "hatches": hatches,
        "bar_width": bar_width,
        "annotations": annotations,
        "bar_pos": bar_pos,
        "color_define": color_define_str,
        "add_plots": add_plots,
        "background_color": background_color,
        "template_file": f"bar_{layout}_{orientation}_{trans_lang}.jinja"
    }

    return context