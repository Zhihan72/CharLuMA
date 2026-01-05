import os
import json
import re
import statistics
import pandas as pd
import numpy as np
from copy import deepcopy
from collections import defaultdict, Counter

from worker.format_worker import *
from worker.types.bar_worker import (
    is_bar_chart,
    infer_bar_subtype,
    extract_bar_chart_data
)
from worker.types.line_worker import (
    is_line_chart, 
    infer_line_subtype, 
    extract_line_chart_data
)
from worker.types.histogram_worker import (
    is_histogram,
    infer_histogram_subtype,
    extract_histogram_chart_data
)
from worker.types.box_worker import (
    is_box_chart,
    infer_box_subtype,
    extract_box_chart_data
)
from worker.types.violin_worker import (
    is_violin_chart,
    infer_violin_subtype,
    extract_violin_chart_data
)
from worker.types.area_worker import (
    is_area_chart,
    infer_area_subtype,
    extract_area_chart_data
)
from worker.types.density_worker import (
    is_density_chart,
    infer_density_subtype,
    extract_density_chart_data
)
from worker.types.scatter_worker import (
    is_scatter_chart,
    infer_scatter_subtype,
    extract_scatter_chart_data
)
from worker.types.radar_worker import (
    is_radar_chart,
    infer_radar_subtype,
    extract_radar_chart_data
)
from worker.types.errorbar_worker import (
    is_errorbar_chart,
    infer_errorbar_subtype,
    extract_errorbar_chart_data
)
from worker.types.errorpoint_worker import (
    is_errorpoint_chart,
    infer_errorpoint_subtype,
    extract_errorpoint_chart_data
)
from worker.types.pie_worker import (
    is_pie_chart,
    infer_pie_subtype,
    extract_pie_chart_data
)
from worker.types.heatmap_worker import (
    is_heatmap_chart,
    infer_heatmap_subtype,
    extract_heatmap_chart_data
)
from worker.types.d3_worker import (
    is_3d_chart
)
from worker.types.contour_worker import (
    is_contour_chart
)
from worker.types.quiver_worker import (
    is_quiver_chart
)

def infer_types(ax_metadata, orig_lang='python'):
    types = []
    if is_bar_chart(ax_metadata, orig_lang): types.append("bar")
    if is_line_chart(ax_metadata, orig_lang): types.append("line")
    if is_area_chart(ax_metadata, orig_lang): types.append("area")
    if is_scatter_chart(ax_metadata, orig_lang): types.append("scatter")
    if is_box_chart(ax_metadata, orig_lang): types.append("box")
    if is_pie_chart(ax_metadata, orig_lang): types.append("pie")
    if is_radar_chart(ax_metadata, orig_lang): types.append("radar")
    if is_heatmap_chart(ax_metadata, orig_lang): types.append("heatmap")
    if is_violin_chart(ax_metadata, orig_lang): types.append("violin")
    if is_density_chart(ax_metadata, orig_lang): types.append("density")
    if is_errorbar_chart(ax_metadata, orig_lang): types.append("errorbar")
    if is_errorpoint_chart(ax_metadata, orig_lang): types.append("errorpoint")
    if is_3d_chart(ax_metadata, orig_lang): types.append("3D")
    if is_contour_chart(ax_metadata, orig_lang): types.append("contour")
    if is_quiver_chart(ax_metadata, orig_lang): types.append("quiver")

    if orig_lang == 'python':
        if set(types) == {'line', 'errorbar'} or set(types) == {'bar', 'errorbar'}:
            types = ['errorbar']

        if set(types) == {'line', 'scatter', 'box'} or set(types) == {'line', 'box'}:
            types = ['box']

        if set(types) == {'scatter', 'violin'}:
            types = ['violin']

        if set(types) == {'violin', 'density'}:
            types = ['density']

        if set(types) == {'line', 'area'}:
            types = ['line']

        if '3D' in types:
            types = ['3D']

    elif orig_lang=='r':

        if types == ['scatter', 'violin']:
            types = ['violin']

        if set(types) == {'bar', 'errorbar'}:
            types = ['errorbar']

        if set(types) == {'line', 'errorpoint'}:
            types = ['errorpoint']

        if set(types) == {'line', 'scatter'} or set(types) == {'line', 'area'} or set(types) == {'line', 'area', 'scatter'}:
            types = ['line']

        if set(types) == {'scatter', 'box'}:
            types = ['box']
        
    return types or []



def infer_orientation(ax_metadata, types, orig_lang):
    type_agnostic = ax_metadata.get("type_agnostic", {})
    obj = ax_metadata.get("object", {})

    if orig_lang == "r" and type_agnostic.get("axis", {}).get("type") == "flip":
        return "horizontal"

    orientation = "vertical"

    if any(t in types for t in ["bar", "box", "violin", "errorbar", "errorpoint"]):
        heights, widths = [], []
        for p in obj.get("patches", []):
            if p.get("object_type") not in ["Rectangle", 'GeomBar', 'GeomCol', 'GeomBoxplot', ]:
                continue
            geom = p.get("geometry", {})
            if "width" in geom and "height" in geom:
                h, w = abs(geom.get("height", 0)), abs(geom.get("width", 0))
                if h > 0 and w > 0:
                    heights.append(h)
                    widths.append(w)

        if heights and widths:
            if len(set(widths)) == 1:
                orientation = "vertical"
            elif len(set(heights)) == 1:
                orientation = "horizontal"
            else:
                try:
                    if statistics.median(widths) > statistics.median(heights):
                        orientation = "horizontal"
                except Exception:
                    pass
    
    x_ticks, y_ticks = type_agnostic.get("x_ticks", []), type_agnostic.get("y_ticks", [])
    if all(isinstance(t.get("text", ""), str) and not is_numeric(t["text"]) for t in x_ticks):
        orientation = "vertical"
    elif all(isinstance(t.get("text", ""), str) and not is_numeric(t["text"]) for t in y_ticks):
        orientation = "horizontal"

    return orientation



def infer_subtypes(metadata, ax_key, orig_lang, trans_lang, types, orientation, tick_labels, tick_pos, annotations):
    if orig_lang=='python' and set(types) in [
        {'bar', 'line', 'scatter'},
        {'bar', 'line'},
        {'bar', 'scatter'},
        {'bar', 'scatter', 'errorbar'},
        {'scatter', 'errorpoint'},
        {'scatter', 'quiver'},
        {'line', 'scatter'},
        {'line', 'density'},
    ]:
        print("Error: Skip! Complex types detected!")
        return None, None
    
    if orig_lang=='r' and set(types) in [
        {'bar', 'line'},
        {'scatter', 'pie'},
        {'scatter', 'violin'},
        {'scatter', 'box', 'violin'},
        {'box', 'violin'},
        {'bar', 'scatter'},
        {'bar', 'line', 'scatter'}
    ]:
        print("Error: Skip! Complex types detected!")
        return None, None
    
    if len(types) > 1:
        print("Warning: Unseen multiple types detected! May cause error.")

    ax_metadata = metadata[ax_key]
    if "bar" in types:
        subtype = infer_bar_subtype(ax_metadata, orientation, orig_lang)
        if not subtype:
            return None, None
        elif subtype=='histogram':
            subtype = infer_histogram_subtype(ax_metadata, orientation, orig_lang)
            if orig_lang=='python':
                try:
                    subtype = "grouped-bar"
                    data = extract_bar_chart_data(metadata, ax_key, orig_lang, trans_lang, types, orientation, tick_labels, tick_pos, annotations, subtype)
                except:
                    subtype = "histogram"
                    data = extract_histogram_chart_data(metadata, ax_key, orig_lang, trans_lang, types, orientation, tick_labels, tick_pos, annotations, subtype)
            else:
                data = extract_histogram_chart_data(metadata, ax_key, orig_lang, trans_lang, types, orientation, tick_labels, tick_pos, annotations, subtype)
        else:
            data = extract_bar_chart_data(metadata, ax_key, orig_lang, trans_lang, types, orientation, tick_labels, tick_pos, annotations, subtype)
        return subtype, data
        
    elif 'line' in types:
        subtype = infer_line_subtype(ax_metadata, orientation, orig_lang)
        if not subtype:
            return None, None
        data = extract_line_chart_data(metadata, ax_key, orig_lang, trans_lang, types, orientation, tick_labels, tick_pos, annotations, subtype)
        return subtype, data

    elif 'area' in types:
        subtype = infer_area_subtype(ax_metadata, orientation, orig_lang)
        if not subtype:
            return None, None
        data = extract_area_chart_data(metadata, ax_key, orig_lang, trans_lang, types, orientation, tick_labels, tick_pos, annotations, subtype)
        return subtype, data

    elif "pie" in types:
        subtype = infer_pie_subtype(ax_metadata, orientation, orig_lang)
        if not subtype:
            return None, None
        data = extract_pie_chart_data(metadata, ax_key, orig_lang, trans_lang, types, orientation, tick_labels, tick_pos, annotations, subtype)
        return subtype, data

    elif "radar" in types:
        subtype = infer_radar_subtype(ax_metadata, orientation, orig_lang)
        if not subtype:
            return None, None
        data = extract_radar_chart_data(metadata, ax_key, orig_lang, trans_lang, types, orientation, tick_labels, tick_pos, annotations, subtype)
        return subtype, data
    
    elif 'box' in types:
        subtype = infer_box_subtype(ax_metadata, orientation, orig_lang)
        if not subtype:
            return None, None
        data = extract_box_chart_data(metadata, ax_key, orig_lang, trans_lang, types, orientation, tick_labels, tick_pos, annotations, subtype)
        return subtype, data

    elif 'errorbar' in types:
        subtype = infer_errorbar_subtype(ax_metadata, orientation, orig_lang)
        if not subtype:
            return None, None
        data = extract_errorbar_chart_data(metadata, ax_key, orig_lang, trans_lang, types, orientation, tick_labels, tick_pos, annotations, subtype)
        return subtype, data

    elif 'errorpoint' in types:
        subtype = infer_errorpoint_subtype(ax_metadata, orientation, orig_lang)
        if not subtype:
            return None, None
        data = extract_errorpoint_chart_data(metadata, ax_key, orig_lang, trans_lang, types, orientation, tick_labels, tick_pos, annotations, subtype)
        return subtype, data

    elif 'density' in types:
        subtype = infer_density_subtype(ax_metadata, orientation, orig_lang)
        if not subtype:
            return None, None
        data = extract_density_chart_data(metadata, ax_key, orig_lang, trans_lang, types, orientation, tick_labels, tick_pos, annotations, subtype)
        return subtype, data

    elif 'violin' in types:
        subtype = infer_violin_subtype(ax_metadata, orientation, orig_lang)
        if not subtype:
            return None, None
        data = extract_violin_chart_data(metadata, ax_key, orig_lang, trans_lang, types, orientation, tick_labels, tick_pos, annotations, subtype)
        return subtype, data

    elif 'heatmap' in types:
        subtype = infer_heatmap_subtype(metadata, ax_key, orientation, orig_lang)
        if not subtype:
            return None, None
        data = extract_heatmap_chart_data(metadata, ax_key, orig_lang, trans_lang, types, orientation, tick_labels, tick_pos, annotations, subtype)
        return subtype, data

    elif 'scatter' in types:
        subtype = infer_scatter_subtype(metadata, ax_key, orientation, orig_lang)
        if not subtype:
            return None, None
        data = extract_scatter_chart_data(metadata, ax_key, orig_lang, trans_lang, types, orientation, tick_labels, tick_pos, annotations, subtype)
        return subtype, data
    
    else:
        return None, None