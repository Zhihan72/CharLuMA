import os
import json
import pandas as pd
import numpy as np
from copy import deepcopy
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import matplotlib.collections as mcollections
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.collections import PathCollection
from matplotlib.transforms import IdentityTransform
from matplotlib.markers import MarkerStyle

import argparse
from pathlib import Path

from worker.format_worker import *
from worker.value_mapping import *

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
        "facecolor": None
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
            "template": [],
        },
        "object": {
            "patches": [],
            "lines": [],
            "collections": [],
            "images": []
        }
    }

def check_twin_axes(fig):
    twin_annotations = {}
    axes = fig.get_axes()

    for i, ax1 in enumerate(axes):
        pos1 = ax1.get_position().bounds
        for j, ax2 in enumerate(axes):
            if i >= j:
                continue

            if ax1.get_shared_x_axes().joined(ax1, ax2):
                twin_annotations[f"ax_{i}"] = {"twin_of": f"ax_{j}", "twin_type": "twinx"}
            if ax1.get_shared_y_axes().joined(ax1, ax2):
                twin_annotations[f"ax_{i}"] = {"twin_of": f"ax_{j}", "twin_type": "twiny"}
    return twin_annotations

def infer_subplot_layout(fig):
    nrows, ncols = 0, 0

    for ax in fig.get_axes():
        try:
            spec = ax.get_subplotspec()
            layout = spec.get_geometry()
            nrows = max(nrows, layout[0])
            ncols = max(ncols, layout[1])
        except Exception:
            continue

    return (nrows, ncols) if nrows > 0 and ncols > 0 else (None, None)

def extract_label_to_color(ax):
    label_to_color = {}
    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        color = None
        if isinstance(handle, ErrorbarContainer):
            color = handle.lines[0].get_color()
        elif isinstance(handle, Line2D):
            color = handle.get_color()
        elif isinstance(handle, Patch):
            color = handle.get_facecolor()
        elif isinstance(handle, PathCollection):
            fc = handle.get_facecolor()
            if isinstance(fc, np.ndarray) and len(fc) > 0:
                color = fc[0]
        label_to_color[label] = rgb_percent_to_hex(color)

    return label_to_color

ref_markers = ['o', 's', '^', 'v', '<', '>', 'D', '*', '+', 'p', 'h', '|', '_']
marker_templates = {
    m: normalize_vertices(MarkerStyle(m).get_path().cleaned().vertices)
    for m in ref_markers
}
def get_marker_shapes(coll):
    if not isinstance(coll, PathCollection):
        return None
    result = []
    for path in coll.get_paths():
        path_vertices = normalize_vertices(path.cleaned().vertices)
        candidates = {
            m: v for m, v in marker_templates.items()
            if v.shape == path_vertices.shape
        }
        best_match = "o"
        min_distance = float('inf')
        for marker, ref_vertices in candidates.items():
            dist = np.linalg.norm(path_vertices - ref_vertices)
            if dist < min_distance:
                min_distance = dist
                best_match = marker
        result.append(best_match)
    return result


def extract_objects_metadata(ax):
    ax_result = {
        "patches": [],
        "lines": [],
        "collections": [],
        "images": []
    }
    label_to_color = {}

    def clean_label(label):
        return label if label and not label.startswith("_") else None
    
    for p in ax.patches:
        obj = {
            "object_type": type(p).__name__,
            "zorder": p.get_zorder(),
            "visible": p.get_visible(),
            "alpha": p.get_alpha(),
            "facecolor": rgb_percent_to_hex(p.get_facecolor()),
            "edgecolor": rgb_percent_to_hex(p.get_edgecolor()),
            "linewidth": p.get_linewidth(),
            "linestyle": p.get_linestyle(),
            "hatch": p.get_hatch(),
            "n_vertices": len(p.get_path().vertices)
        }

        if isinstance(p, mpatches.Rectangle):
            obj.update({
                "geometry": {
                    "x": float(p.get_x()),
                    "y": float(p.get_y()),
                    "width": float(p.get_width()),
                    "height": float(p.get_height())
                }
            })
        elif isinstance(p, mpatches.Wedge):
            obj.update({
                "geometry": {
                    "center": list(p.center),
                    "radius": float(p.r),
                    "theta1": float(p.theta1),
                    "theta2": float(p.theta2)
                }
            })
        elif isinstance(p, mpatches.Circle):
            obj.update({
                "geometry": {
                    "center": list(p.center),
                    "radius": float(p.radius)
                }
            })
        elif isinstance(p, mpatches.Polygon):
            obj.update({
                "geometry": {
                    "xy": [list(map(float, pt)) for pt in p.get_xy()],
                    "closed": p.get_closed()
                }
            })
        elif isinstance(p, mpatches.PathPatch) or isinstance(p, mpatches.Shadow):
            path_disp = p.get_path().transformed(p.get_transform())
            path_data = path_disp.transformed(ax.transData.inverted())
            bbox = path_data.get_extents()
            obj.update({
                "geometry": {
                    "x": float(bbox.x0),
                    "y": float(bbox.y0),
                    "width": float(bbox.width),
                    "height": float(bbox.height),
                    "lower": None,
                    "middle": None,
                    "upper": None,
                }
            })

        label = clean_label(p.get_label())
        if label:
            obj["label"] = label
            label_to_color[label] = obj["facecolor"]
        
        ax_result["patches"].append(obj)
    
    for line in ax.lines:
        obj = {
            "object_type": "Line2D",
            "zorder": line.get_zorder(),
            "visible": line.get_visible(),
            "alpha": line.get_alpha(),
            "color": rgb_percent_to_hex(line.get_color()),
            "linewidth": line.get_linewidth(),
            "linestyle": line.get_linestyle(),
            "marker": line.get_marker(),
            "markerfacecolor": rgb_percent_to_hex(line.get_markerfacecolor()),
            "markeredgecolor": rgb_percent_to_hex(line.get_markeredgecolor()),
            "markersize": line.get_markersize(),
            "geometry": {
                "x": list(line.get_xdata()),
                "y": list(line.get_ydata())
            }
        }

        label = clean_label(line.get_label())
        if label:
            obj["label"] = label
            label_to_color[label] = obj["color"]
        
        ax_result["lines"].append(obj)
    
    for coll in ax.collections:
        obj = {
            "object_type": type(coll).__name__,
            "zorder": coll.get_zorder(),
            "visible": coll.get_visible(),
            "alpha": coll.get_alpha(),
            "facecolors": rgb_percent_to_hex(coll.get_facecolors().tolist()) if hasattr(coll, "get_facecolors") else None,
            "edgecolors": rgb_percent_to_hex(coll.get_edgecolors().tolist()) if hasattr(coll, "get_edgecolors") else None,
            "linewidths": coll.get_linewidths().tolist() if hasattr(coll, "get_linewidths") else None,
            "sizes": coll.get_sizes().tolist() if hasattr(coll, "get_sizes") else None,
            "shape": get_marker_shapes(coll),
            "geometry": coll.get_offsets().tolist() if hasattr(coll, "get_offsets") else None
        }

        if isinstance(coll, PolyCollection):
            try:
                paths = coll.get_paths()
                if paths:
                    vertices = paths[0].vertices
                    obj["geometry"] = {
                        "x": vertices[:, 0].tolist(),
                        "y": vertices[:, 1].tolist()
                    }
            except Exception as e:
                print(f"Warning: failed to extract vertices from PolyCollection: {e}")

        if isinstance(coll, LineCollection):
            try:
                segments = coll.get_segments()
                obj["geometry"] = [
                    {
                        "x": [seg[0][0], seg[1][0]],
                        "y": [seg[0][1], seg[1][1]]
                    }
                    for seg in segments
                ]
            except Exception as e:
                print(f"Warning extracting segments from LineCollection: {e}")

        label = clean_label(coll.get_label())
        if label and obj["facecolors"]:
            obj["label"] = label
            label_to_color[label] = obj["facecolors"][0]
        
        ax_result["collections"].append(obj)
    
    for img in ax.images:
        obj = {
            "object_type": type(img).__name__,
            "zorder": img.get_zorder(),
            "visible": img.get_visible(),
            "alpha": img.get_alpha(),
            "cmap": img.get_cmap().name if img.get_cmap() else None,
            "interpolation": img.get_interpolation(),
        }
        try:
            data = img.get_array()
            if data is not None:
                obj["geometry"] = data.tolist()
        except Exception as e:
            print(f"Warning extracting array from AxesImage: {e}")

        ax_result["images"].append(obj)

    return ax_result, label_to_color

def extract_metadata_from_python_runtime(code_str, label="script.py"):

    try:
        plt.close('all')
        fig = plt.figure()
        exec_globals = {}
        exec(code_str, exec_globals)
        fig = plt.gcf()
        axes = fig.get_axes()
        metadata = default_metadata_template(len(axes))
        metadata["execute"] = {
            "language": "python",
            "source_file": label,
            "executable": True
        }
    except Exception as e:
        print("Error raised: {}".format(e))
        exc_type, exc_value, exc_tb = sys.exc_info()
        metadata = {
            "execute": {
                "language": "python",
                "source_file": label,
                "executable": False,
                "error": {
                    "type": exc_type.__name__,
                    "message": str(exc_value),
                    "line": exc_tb.tb_lineno
                }
            }
        }
        return metadata
    
    if not fig:
        return metadata

    if hasattr(fig, "_suptitle") and fig._suptitle is not None:
        metadata["suptitle"] = fig._suptitle.get_text()
    
    if fig.legends:
        fig_legend = fig.legends[0]
        metadata["legend"] = {
            'exist': True, 
            'loc': fig_legend._loc if hasattr(fig_legend, "_loc") else None,
            'ncol': fig_legend._ncol if hasattr(fig_legend, "_ncol") else None,
            'labels': [text.get_text() for text in fig_legend.get_texts()],
        }
    else:
        metadata["legend"] = {'exist': False, 'loc': None, 'ncol': None, 'labels': None}

    fig_width, fig_height = fig.get_size_inches()
    metadata['plot_size'] =  {
        "width": float(fig_width),
        "height": float(fig_height),
        "unit": "inch"
    }

    metadata["twin_axes"] = check_twin_axes(fig)
    (rows, cols) = infer_subplot_layout(fig)
    metadata["axes_layout"] = {"n_row": rows, "n_col": cols}
    metadata["facecolor"] = rgb_percent_to_hex(fig.get_facecolor())
    
    if not axes:
        return metadata
    
    for k in range(len(axes)):
        ax = axes[k]

        metadata[f'ax_{k}']["type_agnostic"]["axis"] = {
            "position": ax.get_position().bounds,
            "type": ax.name,
            "aspect": ax.get_aspect()
        }
        
        metadata[f'ax_{k}']["type_agnostic"]["title"] = {
            'content': ax.get_title(), 
            'size': ax.title.get_size(), 
            'style': ','.join([ax.title.get_fontproperties().get_weight(),ax.title.get_fontproperties().get_style()]),
        }

        metadata[f'ax_{k}']["type_agnostic"]["x_label"] = {
            'content': ax.get_xlabel(), 
            'size': ax.xaxis.label.get_size(), 
            'style':','.join([ax.xaxis.label.get_fontproperties().get_weight(),ax.xaxis.label.get_fontproperties().get_style()]),
        }

        metadata[f'ax_{k}']["type_agnostic"]["y_label"] = {
            'content':  ax.get_ylabel(), 
            'size': ax.yaxis.label.get_size(), 
            'style': ','.join([ax.yaxis.label.get_fontproperties().get_weight(),ax.yaxis.label.get_fontproperties().get_style()]),
        }
        
        metadata[f'ax_{k}']["type_agnostic"]["background_color"] = rgb_percent_to_hex(ax.get_facecolor())

        xticks = ax.get_xticklabels()
        metadata[f'ax_{k}']["type_agnostic"]["x_ticks"] = [
            {
                "text": tick.get_text(),
                "position": tick.get_position()
            } for tick in xticks
        ]

        yticks = ax.get_yticklabels()
        metadata[f'ax_{k}']["type_agnostic"]["y_ticks"] = [
            {
                "text": tick.get_text(),
                "position": tick.get_position()
            } for tick in yticks
        ]


        legend = ax.get_legend()
        metadata[f'ax_{k}']["type_agnostic"]["legend"] = {
            'exist': True if legend else False, 
            'loc': legend._get_loc() if legend else None, 
            'ncol': legend._ncols if legend else None, 
        }

        spines = ax.spines
        visible_spines = [spine for spine in ['top', 'bottom', 'left', 'right'] if spine in spines and spines[spine].get_visible()]
        metadata[f'ax_{k}']["type_agnostic"]["panel_box"] = (len(visible_spines) == 4)

        texts = ax.texts
        metadata[f'ax_{k}']["type_agnostic"]["annotation"] = []
        for text in texts:
            fontprops = text.get_fontproperties()
            font_size = fontprops.get_size_in_points()
            weight = fontprops.get_weight()
            style = fontprops.get_style()

            annotation_entry = {
                "text": text.get_text(),
                "position": text.xy if hasattr(text, "xy") else None,
                "font_size": fontprops.get_size_in_points(),
                "font_style": f"{fontprops.get_weight()},{fontprops.get_style()}",
                "horizontal_alignment": text.get_ha(),
                "vertical_alignment": text.get_va(),
                "rotation": text.get_rotation(),
                "color": rgb_percent_to_hex(text.get_color()),
            }

            metadata[f'ax_{k}']["type_agnostic"]["annotation"].append(annotation_entry)

        metadata[f'ax_{k}']["type_agnostic"]["grid"] = {
            'x': any(line.get_visible() for line in ax.get_xgridlines()),
            'y': any(line.get_visible() for line in ax.get_ygridlines())
        }

        containers = ax.containers 
        metadata[f'ax_{k}']["type_agnostic"]["container_type"] = [type(c).__name__ for c in containers]

        metadata[f'ax_{k}']["object"], label_to_color = extract_objects_metadata(ax)
        if label_to_color=={}:
            for container in ax.containers:
                label = container.get_label()
                if not label or label.startswith("_"):
                    continue
                facecolors = [
                    rgb_percent_to_hex(obj.get_facecolor())
                    for obj in container
                    if hasattr(obj, "get_facecolor")
                ]
                if facecolors:
                    label_to_color[label] = list(set(facecolors))[0]
        if label_to_color=={}:
            boxes = ax.artists if hasattr(ax, "artists") else ax.patches
            for i, box in enumerate(boxes):
                if i >= len(subgroup_labels):
                    break
                color = rgb_percent_to_hex(box.get_facecolor())
                label = subgroup_labels[i % len(subgroup_labels)]
                label_to_color[label] = color
        if label_to_color=={}:
            label_to_color = extract_label_to_color(ax)
        if len(label_to_color.keys()) < 2:
            label_to_color = {}
        metadata[f'ax_{k}']["type_agnostic"]["label_to_color"] = label_to_color

    return metadata

if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Extract plot metadata from Python scripts.")
    p.add_argument("--head_dir")
    p.add_argument("--suffix")
    args = p.parse_args()

    scripts = [f for f in os.listdir(args.head_dir) if f.endswith(args.suffix)]
    for script_pt in scripts:
        script_path = os.path.join(args.head_dir, script_pt)

        with open(script_path, "r") as f:
            code_str = f.read()
        print("Processing: {}".format(os.path.basename(script_path)))
        try:
            metadata = extract_metadata_from_python_runtime(code_str, label=script_path)
        except Exception as e:
            print("Exception in {}: {}".format(script_pt, e))
            metadata = {}
        save_path = script_path.split('.')[0] + '.json'
        json_string = json.dumps(metadata, indent=4, default=str)
        with open(save_path, 'w') as fp:
            fp.write(json_string)
        print("Saving to: {}".format(os.path.basename(save_path)))