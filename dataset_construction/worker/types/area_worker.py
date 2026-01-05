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

def is_area_chart(ax_metadata, orig_lang):
    if orig_lang == "python":
        collections = ax_metadata.get("object", {}).get("collections", [])
        for coll in collections:
            if coll.get("object_type") == "PolyCollection":
                x = coll.get("geometry", {}).get("x", [])
                y = coll.get("geometry", {}).get("y", [])

                if not x or not y:
                    continue
                
                if all(val<1.0 for val in y):
                    continue
                
                bool_y_zero = [val==0 for val in y]
                if sum(bool_y_zero) > len(bool_y_zero) * 0.3:
                    return True

    elif orig_lang == "r":
        lines = ax_metadata.get("object", {}).get("lines", [])
        if any(l.get("object_type") in ["GeomDensity",] for l in lines):
            return False
        return any(l.get("object_type") in ["GeomArea",] for l in lines)


def infer_area_subtype(ax_metadata, orientation, orig_lang):
    obj = ax_metadata.get("object", {})
    collections = obj.get("collections", [])
    lines = obj.get("lines", [])

    if orig_lang == "python":
        is_multi = len(collections) > 1
        is_stacked = False
        for coll in collections:
            if coll.get("object_type", None) not in ["PolyCollection",]:
                continue
            y = coll.get("geometry", {}).get("y", [])
            bool_y_zero = [val==0 for val in y]
            if sum(bool_y_zero)==0:
                is_stacked = True
                break
    elif orig_lang == "r":
        is_multi = len(lines) > 1
        is_stacked = False
        for l in lines:
            if l.get("position", "") in ["PositionStack"]:
                is_stacked = True
        
    if is_multi and is_stacked:
        return "stacked-area"
    elif is_multi and not is_stacked:
        return "multi-area"
    else:
        return "base-area"


def extract_area_chart_data(metadata, ax_key, orig_lang, trans_lang, types, orientation, tick_labels, tick_pos, annotations, subtype):
    ax_metadata = metadata[ax_key]
    obj = ax_metadata.get("object", {})
    
    label_to_color = ax_metadata.get("type_agnostic", {}).get("label_to_color", [])
    group_labels = list(label_to_color.keys()) if label_to_color else []

    series = defaultdict(lambda: {
        "x": [],
        "y_low": [],
        "y_high": [],
        "color": [],
        "alpha": []
    })

    if orig_lang == "python":
        collections = [coll for coll in obj.get("collections", []) if coll.get("object_type", None) in ["PolyCollection",]]
        for k in range(len(collections)):
            coll = collections[k]
            facecolors = coll.get("facecolors", [None])
            alpha = coll.get("alpha", {})
            if facecolors!=[]:
                color = rgb_percent_to_hex(facecolors[0])
                label = find_label_by_color(color, label_to_color)
            else:
                color = ""
                label = f'group_{k}'
            
            x_coll = coll.get("geometry", {}).get("x", [])
            y_coll = coll.get("geometry", {}).get("y", [])
            x, y_low, y_high = split_polygon_fill(x_coll, y_coll)

            series[label]["x"] = x
            series[label]["y_low"] = y_low
            series[label]["y_high"] = y_high
            series[label]["color"] = color
            series[label]["alpha"] = alpha

    elif orig_lang=='r':
        lines = [l for l in obj.get("lines", []) if l.get("object_type", None)  in ["GeomArea",]]
        
        for k in range(len(lines)):
            line = lines[k]
            color = rgb_percent_to_hex(line.get("color", None))
            alpha = line.get("alpha", {})
            label = find_label_by_color(color, label_to_color)
            if not label:
                label = f'group_{k}'
            x = line.get("geometry", {}).get("x", [])
            y = line.get("geometry", {}).get("y", [])

            idx_int = [i for i in range(len(x)) if type(x[i])==int]
            x = [x[i] for i in range(len(x)) if i in idx_int]
            y = [y[i] for i in range(len(y)) if i in idx_int]

            series[label]["x"] = x
            series[label]["y_low"] = y
            series[label]["y_high"] = y
            series[label]["color"] = color
            series[label]["alpha"] = alpha
    
    group_labels = list(series.keys())
    x_values = [s["x"] for s in series.values()]
    y_values = [s["y_low"] for s in series.values()]
    y_high_values = [s["y_high"] for s in series.values()]
    colors = [s["color"] for s in series.values()]
    alphas = [s["alpha"] for s in series.values()]

    if 'stacked' in subtype:
        y_values = []
        for k in range(len(y_high_values)):
            if orig_lang == "python":
                if k==0:
                    y_values.append(y_high_values[k])
                else:
                    y_values.append([y_high_values[k][i] - y_high_values[k-1][i] for i in range(len(y_high_values[k]))])
            elif orig_lang == "r":
                if k==len(y_high_values)-1:
                    y_values.append(y_high_values[k])
                else:
                    y_values.append([y_high_values[k][i] - y_high_values[k+1][i] for i in range(len(y_high_values[k]))])
    
    background_color = rgb_percent_to_hex(ax_metadata["type_agnostic"].get("background_color") or '#Ffffff')
    if trans_lang=='latex':
        color_define_str, color_labels = generate_latex_color_define(colors, background_color, ax_key)
        colors = color_labels
        background_color = 'cb'

    template = "area"
    x_values = x_values[0]
    alphas = alphas[0]
    if len(group_labels)==1:
        y_values = y_values[0]
        colors = colors[0]
        template += '_base'
    else:
        if 'stacked' in subtype:
            template += '_stacked'
        else:
            template += '_multi'
    template += f'_{trans_lang}.jinja'

    if len(tick_labels)!=len(x_values):
        has_x_str = False
    else:
        has_x_str = True

    if (trans_lang=='r' or orig_lang=='r') and 'stacked' in subtype:
        group_labels = group_labels[::-1]
        y_values = y_values[::-1]
        colors = colors[::-1]

    if trans_lang=='latex' and 'stacked' in subtype:
        y_values_stacked = []
        base = [0.0] * len(y_values[0])
        for row in y_values:
            cumulative_row = [base[j] + row[j] for j in range(len(row))]
            y_values_stacked.append(cumulative_row)
            base = cumulative_row
        y_values = y_values_stacked

    if trans_lang!='latex':
        x_values = format_list_for_lang(x_values, trans_lang, "num")
        y_values = format_list_for_lang(y_values, trans_lang, "num")
        if type(colors)==list:
            colors = format_list_for_lang(colors, trans_lang, "str")
        if len(group_labels)>1:
            group_labels = format_list_for_lang(group_labels, trans_lang, "str")
        color_define_str = ""
        x_pos_str = ""
    else:
        group_labels = ["{"+escape_string(item, trans_lang)+"}" for item in group_labels if item!=None]
        x_pos_str = format_list_for_lang(x_values, trans_lang, "num")
    
    context = {
        "group_labels": group_labels,
        "x_values": x_values,
        "y_values": y_values,
        "colors": colors,
        "alphas": alphas,
        "has_x_str": has_x_str,
        "background_color": background_color,
        "color_define": color_define_str,
        "x_pos_str": x_pos_str,
        "template_file": template
    }

    return context