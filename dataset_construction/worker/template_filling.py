import os
import json
import math
from jinja2 import Template, Environment, FileSystemLoader
from collections import defaultdict, OrderedDict

import argparse
from pathlib import Path

from worker.type_worker import infer_orientation, infer_types, infer_subtypes
from worker.value_mapping import *
from worker.format_worker import *

def render_plot_script(template_dir, metadata_dir, metadata_file, trans_lang):

    if 'chartcoder_160k' in metadata_dir:
        orig_lang = 'python'
    elif 'r_40k' in metadata_dir:
        orig_lang = 'r'
    elif "tikz" in metadata_dir:
        orig_lang = 'latex'
    
    if trans_lang=='python':
        trans_suffix = "_python.py"
    elif trans_lang=='r':
        trans_suffix = "_r.R"
    elif trans_lang=='latex':
        trans_suffix = "_latex.tex"
    
    with open(os.path.join(metadata_dir, metadata_file), "r") as f:
        metadata = json.load(f)
    
    ax_key_list = [k for k in metadata.keys() if 'ax_' in k]
    if len(ax_key_list)==0:
        raise ValueError("Empty metadata dictionary!")

    plot_width, plot_height = 0, 0
    if metadata["plot_size"].get("unit")=="inch":
        plot_width = metadata["plot_size"].get("width") or 6.4
        plot_height = metadata["plot_size"].get("height") or 4.8
    if plot_width <= 3 and plot_height<=3:
        plot_width = 6.4
        plot_height = 4.8

    contexts = []
    empty_templates = []

    for key in ax_key_list:

        types = infer_types(metadata[key], orig_lang)
        orientation = infer_orientation(metadata[key], types, orig_lang)
        metadata[key]["type_specific"]["type"] = types
        metadata[key]["type_specific"]["orientation"] = orientation
        print(f"\nProcessing metadata: {metadata_file}")

        has_legend = metadata[key]["type_agnostic"]["legend"].get("exist") or False
        legend_loc = metadata[key]["type_agnostic"]["legend"].get("loc")
        legend_ncols = metadata[key]["type_agnostic"]["legend"].get("ncol")
        grid_info = metadata[key]["type_agnostic"].get("grid", {})
        has_grid =  grid_info.get("x", False) or grid_info.get("y", False)
        has_panel_box = metadata[key]["type_agnostic"].get("panel_box") or False
        x_label_info = metadata[key]["type_agnostic"].get("x_label")
        y_label_info = metadata[key]["type_agnostic"].get("y_label")
        title_info = metadata[key]["type_agnostic"].get("title")
        x_label = x_label_info["content"]
        x_font_size = x_label_info["size"]
        x_font_weight_style = x_label_info.get("style") or ""
        y_label = y_label_info["content"]
        y_font_size = y_label_info["size"]
        y_font_weight_style = y_label_info.get("style") or ""
        title = title_info["content"]
        title_font_size = title_info["size"]
        title_font_weight_style = title_info.get("style") or ""
        x_ticks = metadata[key]["type_agnostic"].get("x_ticks")
        y_ticks = metadata[key]["type_agnostic"].get("y_ticks")
        x_tick_labels, x_tick_pos = extract_sorted_ticks(x_ticks, "x")
        y_tick_labels, y_tick_pos = extract_sorted_ticks(y_ticks, "y")
        annotation_list = metadata[key]["type_agnostic"].get("annotation")

        if orientation=='vertical':
            tick_labels = x_tick_labels
            tick_pos = x_tick_pos
            minor_tick_labels = y_tick_labels
            annotations, _ = extract_sorted_ticks(annotation_list, "x")
        else:
            tick_labels = y_tick_labels
            tick_pos = y_tick_pos
            minor_tick_labels = x_tick_labels
            annotations, _ = extract_sorted_ticks(annotation_list, "y")
        
        minor_tick_labels = [val.replace('−', '-') for val in minor_tick_labels]
        if not is_all_none(minor_tick_labels) and is_numeric_tick_list(minor_tick_labels):
            has_tick = True
            valid_pairs = [(float(x), x) for x in minor_tick_labels if x is not None]
            valid_pairs.sort(key=lambda pair: pair[0])
            sec_ticks, sec_tick_labels = zip(*valid_pairs)
            sec_ticks = list(sec_ticks)
            sec_tick_labels = list(sec_tick_labels)
            if len(sec_ticks) > 1:
                sec_gap = sec_ticks[1] - sec_ticks[0]
                sec_limit_low = sec_ticks[0]
                sec_limit_high = sec_ticks[-1]
            else:
                sec_gap, sec_limit_low, sec_limit_high = 0, 0, 0
        else:
            has_tick = False
            sec_ticks = []
            sec_tick_labels = []
            sec_gap, sec_limit_low, sec_limit_high = 0, 0, 0
        
        has_annot = is_all_none(annotations)==False
        if has_annot:
            annotation_weight_style = annotation_list[0]["font_style"]
            annotation_ha = annotation_list[0]["horizontal_alignment"]
            annotation_va = annotation_list[0]["vertical_alignment"]
        else:
            annotation_weight_style, annotation_ha, annotation_va = None, None, None

        try:
            subtype, type_data = infer_subtypes(metadata, key, orig_lang, trans_lang, types, orientation, tick_labels, tick_pos, annotations)
            print("Type of {}: {} - {}".format(key, types, subtype))
        except Exception as e:
            print(e)
            continue

        if not subtype:
            continue

        template_file = type_data["template_file"]
        metadata[key]["type_specific"]["sub_type"] = subtype
        if template_file not in metadata[key]["type_specific"]["template"]:
            metadata[key]["type_specific"]["template"].append(template_file)
        print("Template of {}: {}".format(key, template_file))

        title = escape_string(title, trans_lang)
        x_label = escape_string(x_label, trans_lang)
        y_label = escape_string(y_label, trans_lang)
        x_font_weight, x_font_style, x_fontface = map_font(x_font_weight_style, trans_lang)
        y_font_weight, y_font_style, y_fontface = map_font(y_font_weight_style, trans_lang)
        title_font_weight, title_font_style, title_fontface = map_font(title_font_weight_style, trans_lang)
        annotation_weight, annotation_style, annotation_fontface = map_font(annotation_weight_style, trans_lang)
        annotation_ha = map_annot_hjust(annotation_ha, orig_lang, trans_lang)
        annotation_va = map_annot_vjust(annotation_va, orig_lang, trans_lang)
        legend_loc, legend_ncols = map_legend(legend_loc, legend_ncols, orig_lang, trans_lang) if has_legend else (None, None)

        x_font_size = map_font_size(x_font_size, trans_lang)
        y_font_size = map_font_size(y_font_size, trans_lang)
        title_font_size = map_font_size(title_font_size, trans_lang)

        tick_labels = format_list_for_lang(tick_labels, trans_lang, "str")
        tick_pos = format_list_for_lang(tick_pos, trans_lang, "num")
        sec_ticks = format_list_for_lang(sec_ticks, trans_lang, "num")
        sec_tick_labels = format_list_for_lang(sec_tick_labels, trans_lang, "str")
        annotations = format_list_for_lang(annotations, trans_lang, "str")
        
        context = {
            "width": plot_width,
            "height": plot_height,
            "has_grid": has_grid,
            "has_panel_box": has_panel_box,
            "has_legend": has_legend,
            "legend_loc": legend_loc,
            "legend_ncols": legend_ncols,
            "has_annot": has_annot,
            "annotations": annotations,
            "annotation_ha": annotation_ha,
            "annotation_va": annotation_va,
            "title": title,
            "title_font_size": title_font_size,
            "title_font_weight": title_font_weight,
            "title_font_style": title_font_style,
            "title_font_face": title_fontface,
            "x_label": x_label,
            "x_font_size": x_font_size,
            "x_font_weight": x_font_weight,
            "x_font_style": x_font_style,
            "x_font_face": x_fontface,
            "y_label": y_label,
            "y_font_size": y_font_size,
            "y_font_weight": y_font_weight,
            "y_font_style": y_font_style,
            "y_font_face": y_fontface,
            "annotation_weight": annotation_weight,
            "annotation_style": annotation_style,
            "annotation_fontface": annotation_fontface,
            "has_tick": has_tick,
            "sec_ticks": sec_ticks,
            "sec_tick_labels": sec_tick_labels,
            "sec_gap": sec_gap,
            "sec_limit_low": sec_limit_low,
            "sec_limit_high": sec_limit_high,
            "tick_labels": tick_labels,
            "tick_pos": tick_pos
        }
        context.update(type_data)

        contexts.append(context)
        empty_templates.append(template_file)
    

    json_string = json.dumps(metadata, indent=4, default=str)
    with open(os.path.join(metadata_dir, metadata_file), "w") as f:
        f.write(json_string)
    
    if len(empty_templates)==0:
        raise ValueError("No templates or contexts extracted!")

    elif len(empty_templates)==1:
        context = contexts[0]
        template_file = empty_templates[0]
        with open(os.path.join(template_dir, template_file)) as f:
            env = Environment(
                loader=FileSystemLoader("templates"),
                trim_blocks=True,
                lstrip_blocks=True
            )
            template = env.from_string(f.read())
        rendered_code = template.render(**context)

    else:
        n_col = int(math.ceil(math.sqrt(len(contexts))))
        n_row = int(math.ceil(len(contexts) / n_col))
        if len(contexts) <= 3:
            if orig_lang=='r':
                n_col, n_row = len(contexts), 1
            else:
                n_col, n_row = 1, len(contexts)
        axes_layout = metadata["axes_layout"]
        if axes_layout:
            if "n_row" in axes_layout and "n_col" in axes_layout:
                n_row = axes_layout["n_row"]
                n_col = axes_layout["n_col"]
        subplot_pos_py = subplot_positions_py(n_row, n_col)

        is_polar = any("radar" in f for f in empty_templates)

        axes_codes = []
        import_packages = []
        color_define = []
        for k in range(len(contexts)):
            context = contexts[k]
            template_file = empty_templates[k]
            with open(os.path.join(template_dir, template_file)) as f:
                env = Environment(
                    loader=FileSystemLoader("templates"),
                    trim_blocks=True,
                    lstrip_blocks=True
                )
                template = env.from_string(f.read())
            filled_code = template.render(**context)

            code_lines = []
            if trans_lang=='python':
                for l in filled_code.split("\n"):
                    if 'plt.subplots' in l or 'plt.tight_layout()' in l or 'plt.show()' in l:
                        continue
                    if 'ax.' in l:
                        l = l.replace('ax.', f'axes[{subplot_pos_py[k]}].')
                    if "import" in l:
                        import_packages.append(l)
                    else:
                        code_lines.append(l)
            elif trans_lang=='r':
                if "radarchart(" in filled_code:
                    library_lines, axis_lines= split_r_radarchart(filled_code, k)
                    import_packages.extend(library_lines.split('\n'))
                    code_lines = axis_lines.split('\n')
                else:
                    for l in filled_code.split("\n"):
                        if 'print(p)' in l:
                            continue
                        if l.strip().startswith('p <- p +'):
                            l = l.replace('p <- p +', f'p{k} <- p{k} +')
                        elif l.strip().startswith('p <-'):
                            l = l.replace('p <-', f'p{k} <-')
                        if "library(" in l:
                            import_packages.append(l)
                        else:
                            code_lines.append(l)
            elif trans_lang=='latex':
                library_lines, color_lines, axis_lines= split_tex_simple(filled_code, is_polar)
                import_packages.extend(library_lines.split('\n'))
                color_define.extend(color_lines.split('\n'))
                code_lines = axis_lines.split('\n')
            axes_codes.append('\n'.join(code_lines))
        
        if is_polar and trans_lang=='latex':
            axes_codes = '&\n'.join(axes_codes)
        else:
            axes_codes = '\n\n'.join(axes_codes)
        import_packages = '\n'.join(list(set(import_packages)))
        color_define = '\n'.join(color_define)
        plot_objects = ','.join([f'p{k}' for k in range(len(contexts))])


        subplot_context = {
            "import_packages": import_packages,
            "n_row": n_row,
            "n_col": n_col,
            "width": plot_width,
            "height": plot_height,
            "axes_codes": axes_codes,
            "plot_objects": plot_objects,
            "color_define": color_define,
            "is_polar": is_polar,
            "suptitle": metadata["suptitle"]
        }
        with open(os.path.join(template_dir, f"subplots_{trans_lang}.jinja")) as f:
            env = Environment(
                loader=FileSystemLoader("templates"),
                trim_blocks=True,
                lstrip_blocks=True
            )
            template = env.from_string(f.read())
        rendered_code = template.render(**subplot_context)
    
    output_file = metadata_file.replace('_object.json', f'_object{trans_suffix}')
    if trans_lang=='python':
        fig_save_line = '\nplt.savefig("{}")'.format(os.path.join(metadata_dir, output_file.split('.')[0]+'.jpg'))
        rendered_code += fig_save_line
    elif trans_lang=='r':
        fig_save_line = 'jpeg("{}", width = {}, height = {}, units = "in", res = 100)\n'.format(os.path.join(metadata_dir, output_file.split('.')[0]+'.jpg'), plot_width, plot_height)
        rendered_code = fig_save_line + rendered_code
    with open(os.path.join(metadata_dir, output_file), "w") as f:
        f.write(rendered_code)
    
    print(f"Generated plot script: {output_file}")


if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Extract plot metadata from Python scripts.")
    p.add_argument("--head_dir", help="Directory containing .py scripts OR a single .py file")
    p.add_argument("--trans_langs", nargs="+", help="Directory containing .py scripts OR a single .py file")
    args = p.parse_args()

    metadata_dir = args.head_dir
    trans_langs = args.trans_langs
    template_dir = "./template"
    metadata_files = sorted([f for f in os.listdir(metadata_dir) if f.endswith('_object.json')])
    
    for trans_lang in trans_langs:

        for meta_fl in metadata_files:

            try:
                render_plot_script(
                    template_dir=template_dir,
                    metadata_dir=metadata_dir,
                    metadata_file=meta_fl,
                    trans_lang=trans_lang,
                )
            except Exception as e:
                print("❌ {}".format(e))
                continue