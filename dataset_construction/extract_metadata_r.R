library(ggplot2)
library(grid)
library(gtable)
library(jsonlite)
library(ggpattern)
library(R.utils)

`%||%` <- function(a, b) {
  if (length(a) > 0 && !is.null(a[1]) && !is.na(a[1])) a else b
}

default_metadata_template <- function(num_ax) {
  metadata <- list(
    execute = list(
      source_file = NA,
      language = NA,
      executable = NA
    ),
    suptitle = NA,
    legend = NA,
    plot_size = list(width = 6.4, height = 4.8, unit = "inch"),
    twin_axes = list(),
    axes_layout = list(),
    facecolor = NA
  )

  for (k in seq_len(num_ax) - 1) {
    metadata[[paste0("ax_", k)]] <- default_metadata_ax()
  }

  return(metadata)
}

default_metadata_ax <- function() {
  list(
    type_agnostic = list(
      axis = list(),
      facet_wrap = FALSE,
      title = list(),
      x_label = list(),
      y_label = list(),
      x_ticks = list(),
      y_ticks = list(),
      legend = list(),
      grid = list(),
      panel_box = FALSE,
      background_color = NA,
      colorbar = list(),
      annotation = list(),
      label_to_color = list(),
      container_type = list()
    ),
    type_specific = list(
      type = NA,
      sub_type = NA,
      orientation = NA,
      template = list()
    ),
    object = list(
      patches = list(),
      lines = list(),
      collections = list(),
      segments = list()
    )
  )
}


extract_objects_metadata <- function(p) {
  result <- list(
    patches = list(),
    lines = list(),
    collections = list(),
    texts = list(),
    segments = list()
  )

  pb <- ggplot2::ggplot_build(p)

  for (i in seq_along(p$layers)) {
    layer <- p$layers[[i]]
    stat_type <- class(layer$stat)[1]
    geom_type <- class(layer$geom)[1]
    data <- pb$data[[i]]

    if (is.data.frame(data) && nrow(data) > 0) {
      groups <- split(data, data$group)

      if (geom_type %in% c("GeomBar", "GeomCol", "GeomRect", "GeomBoxplot", "GeomViolin",
                           "GeomPolygon", "GeomHistogram", "GeomTile", "GeomRug", "GeomRaster")) {
        if (geom_type == "GeomBar" && stat_type == "StatBin") {
          geom_type <- "GeomHistogram"
        }
        for (g in groups) {
          for (j in seq_len(nrow(g))) {
            patch_geom <- list(
              object_type = geom_type,
              zorder = g$PANEL[j],
              visible = TRUE,
              alpha = g$alpha[j],
              facecolor = g$fill[j] %||% NA,
              edgecolor = g$colour[j] %||% NA,
              linewidth = g$linewidth[j] %||% NA,
              linestyle = g$linetype[j] %||% NA,
              hatch = g$pattern[j] %||% NA,
              geometry = list()
            )
            if (geom_type %in% c("GeomBar", "GeomCol", "GeomHistogram", "GeomRaster", "GeomRect")) {
              patch_geom$geometry <- list(
                x = g$xmin[j],
                y = g$ymin[j],
                width = g$xmax[j] - g$xmin[j],
                height = g$ymax[j] - g$ymin[j]
              )
            } else if (geom_type == "GeomBoxplot") {
              patch_geom$geometry <- list(
                x = g$xmin[j],
                y = g$ymin[j],
                width = g$xmax[j] - g$xmin[j],
                height= g$ymax[j] - g$ymin[j],
                lower = g$lower[j],
                middle = g$middle[j],
                upper = g$upper[j],
                is_notched = g$notch[j]
              )
            } else if (geom_type == "GeomViolin") {
              patch_geom$geometry <- list(
                x = g$x[j],
                y = g$y[j],
                width = g$violinwidth[j],
                area = g$density[j]
              )
            } else if (geom_type == "GeomPolygon") {
              patch_geom$geometry <- list(
                x = g$x[j],
                y = g$y[j],
                group = g$group[j]
              )
            } else if (geom_type == "GeomRug") {
              patch_geom$geometry <- list(
                x = g$x[j],
                y = g$y[j],
                side = g$side[j] %||% NA
              )
            } else if (geom_type == "GeomTile") {
              patch_geom$geometry <- list(
                x = g$xmin[j],
                y = g$ymin[j],
                data = g$value[j],
                width = g$xmax[j] - g$xmin[j],
                height = g$ymax[j] - g$ymin[j]
              )
            }
            result$patches[[length(result$patches) + 1]] <- patch_geom
          }
        }
      }

      else if (geom_type %in% c("GeomLine", "GeomPath", "GeomArea", "GeomDensity",
                                "GeomSmooth", "GeomRibbon", "GeomHline", "GeomVline", "GeomContour")) {
        for (g in groups) {
          line_geom <- list(
            object_type = geom_type,
            zorder = g$PANEL[1],
            visible = TRUE,
            alpha = g$alpha[1],
            color = g$colour[1] %||% g$fill[1] %||% NA,
            linewidth = g$linewidth[1],
            linestyle = g$linetype[1]
          )

          if (geom_type == "GeomRibbon") {
            line_geom$geometry <- list(x = g$x, ymin = g$ymin, ymax = g$ymax)
          } else if (geom_type == "GeomHline") {
            line_geom$geometry <- list(yintercept = g$yintercept)
          } else if (geom_type == "GeomVline") {
            line_geom$geometry <- list(xintercept = g$xintercept)
          } else if (geom_type == "GeomContour") {
            message("Warning: GeomContour happens. This object is not well extracted!")
            line_geom$geometry <- list(x = g$x, y = g$y)
          } else {
            line_geom$geometry <- list(x = g$x, y = g$y)
          }

          if (geom_type == "GeomArea"){
            line_geom$position <- class(layer$position)[1] %||% NA
            line_geom$color <- g$fill[1] %||% NA
          }

          if (geom_type == "GeomDensity"){
            line_geom$color <- g$colour[1] %||% NA
            line_geom$fill <- g$fill[1] %||% NA
          }

          result$lines[[length(result$lines) + 1]] <- line_geom
        }
      }

      else if (geom_type %in% c("GeomSegment", "GeomCurve", "GeomErrorbar")) {
        for (g in groups) {
          for (j in seq_len(nrow(g))) {
            geom_data <- list(
              object_type = geom_type,
              zorder = g$PANEL[j],
              visible = TRUE,
              alpha = g$alpha[j],
              color = g$colour[j],
              linewidth = g$linewidth[j],
              linestyle = g$linetype[j],
              geometry = list()
            )
            if (geom_type %in% c("GeomSegment", "GeomCurve")) {
              geom_data$geometry <- list(
                x = g$x[j], y = g$y[j],
                xend = g$xend[j], yend = g$yend[j]
              )
            } else if (geom_type == "GeomErrorbar") {
              geom_data$geometry <- list(
                x = g$x[j],
                ymin = g$ymin[j],
                ymax = g$ymax[j],
                width = g$width[j] %||% NA
              )
            }
            result$segments[[length(result$segments) + 1]] <- geom_data
          }
        }
      }

      else if (geom_type %in% c("GeomPoint", "GeomJitter", "GeomDotplot")) {
        for (g in groups) {
          result$collections[[length(result$collections) + 1]] <- list(
            object_type = geom_type,
            zorder = g$PANEL[1],
            visible = TRUE,
            alpha = g$alpha,
            facecolors = g$fill,
            edgecolors = g$colour,
            linewidths = g$stroke %||% g$linewidth %||% NA,
            sizes = g$size,
            shape = g$shape %||% g$pch %||% NA,
            geometry = mapply(function(x, y) list(x, y), g$x, g$y, SIMPLIFY = FALSE)
          )
        }
      }

      else if (geom_type == "GeomText") {
        for (g in groups) {
          for (j in seq_len(nrow(g))) {
            result$texts[[length(result$texts) + 1]] <- list(
              object_type = "GeomText",
              zorder = g$PANEL[j],
              visible = TRUE,
              text = g$label[j],
              position = list(x = g$x[j], y = g$y[j]),
              font_size = g$size[j],
              color = g$colour[j],
              angle = g$angle[j] %||% 0,
              hjust = g$hjust[j] %||% NA,
              vjust = g$vjust[j] %||% NA
            )
          }
        }
      }
    }
  }

  return(result)
}

get_label_to_color <- function(p) {
  built <- ggplot2::ggplot_build(p)
  aesthetics_to_try <- c("fill", "colour")

  for (aesthetic in aesthetics_to_try) {
    aes_mapping <- p$mapping[[aesthetic]]
    aes_expr <- if (!is.null(aes_mapping)) rlang::as_label(rlang::get_expr(aes_mapping)) else NULL

    layer_data <- built$data[[1]]
    aes_values <- NULL
    if (!is.null(aes_expr) && aes_expr %in% names(p$data)) {
      aes_values <- unique(as.character(p$data[[aes_expr]]))
    } else if (!is.null(aes_expr) && aes_expr %in% names(layer_data)) {
      aes_values <- unique(as.character(layer_data[[aes_expr]]))
    } else {
      candidates <- intersect(names(layer_data), aesthetics_to_try)
      if (length(candidates) > 0) {
        aes_expr <- candidates[1]
        aes_values <- unique(as.character(layer_data[[aes_expr]]))
      }
    }

    scale_obj <- built$plot$scales$get_scales(aesthetic)

    if (!is.null(scale_obj) && !is.null(aes_values)) {
      tryCatch({
        color_values <- scale_obj$map(aes_values)
        label_to_color <- as.list(setNames(color_values, aes_values))
        return(label_to_color)
      }, error = function(e) {
        message("⚠️ Failed to map scale for aesthetic '", aesthetic, "': ", conditionMessage(e))
      })
    }
  }

  warning("No valid fill or colour aesthetic found in plot.")
  return(list())
}


get_annotations <- function(texts) {
  annotations <- list()
  for (i in seq_along(texts)) {
    t <- texts[[i]]
    annotations[[length(annotations) + 1]] <- list(
      text = t$text %||% NA,
      position = t$position %||% NA,
      font_size = t$font_size %||% NA,
      font_style = NA,
      horizontal_alignment = t$hjust %||% NA,
      vertical_alignment = t$vjust %||% NA,
      rotation = t$angle %||% NA,
      color = t$color %||% NA
    )
  }
  return(annotations)
}

has_colorbar <- function(p) {
  color_scale <- p$scales$get_scales("colour")
  if (!inherits(color_scale, "ScaleContinuous")) {
    return(FALSE)
  } else {
    return(TRUE)
  }
}

safe_eval_all_plots <- function(code_str, env = new.env()) {
  exprs <- parse(text = code_str)
  plots <- list()

  for (e in exprs) {
    result <- tryCatch(eval(e, envir = env), error = function(err) NA)

    if (inherits(result, "ggplot")) {
      is_duplicate <- any(vapply(plots, function(p) identical(p, result), logical(1)))
      if (!is_duplicate) {
        plots[[length(plots) + 1]] <- result
      }
    }
  }

  if (length(plots) == 0) {
    last_plot <- tryCatch(ggplot2::last_plot(), error = function(e) NA)
    if (inherits(last_plot, "ggplot")) {
      plots[[1]] <- last_plot
    }
  }

  if (length(plots) == 0) {
    warning("Not a ggplot object. Attempting base R barplot extraction.")
    return(NA)
  }

  return(plots)
}

remove_misleading_lines <- function(code_str) {
  lines <- unlist(strsplit(code_str, "\n"))
  filtered <- lines[!grepl("jpeg\\s*\\(|dev\\.off\\s*\\(", lines)]
  paste(filtered, collapse = "\n")
}

resolve_text_size <- function(size_obj, base_size) {
  if (inherits(size_obj, "rel")) {
    return(as.numeric(size_obj) * base_size)
  } else if (!is.null(size_obj)) {
    return(as.numeric(size_obj))
  } else {
    return(as.numeric(base_size))
  }
}

extract_metadata_from_r_runtime <- function(code_str, label = "script.R") {

  result <- tryCatch({
    clean_code_str <- remove_misleading_lines(code_str)
    plot_list <- safe_eval_all_plots(clean_code_str)
    if (is.null(plot_list)) return(NA)

    metadata <- default_metadata_template(length(plot_list))

    metadata$execute <- list(
      source_file = label,
      language = "r",
      executable = TRUE
    )

    dev_size <- tryCatch(par("din"), error = function(e) NULL)
    if (!is.null(dev_size)) {
      metadata$plot_size <- list(
        width = dev_size[1],
        height = dev_size[2],
        unit = "inch"
      )
    }

    for (i in seq_along(plot_list)) {
      p <- plot_list[[i]]
      pb <- ggplot_build(p)
      gt <- ggplotGrob(p)
      theme <- p$theme

      safe_theme_get <- function(x, field) {
        if (!is.null(x) && !is.null(x[[field]])) return(x[[field]]) else return(NA)
      }

      if (i==1 && is.null(dev_size)){
        width_inch <- convertWidth(sum(gt$widths), "in", valueOnly = TRUE)
        height_inch <- convertHeight(sum(gt$heights), "in", valueOnly = TRUE)
        metadata$plot_size <- list(
          width = width_inch,
          height = height_inch,
          unit = "inch"
        )
      }

      ax_key <- paste0("ax_", i - 1)
      coord_class <- class(p$coordinates)[1]
      coord_name <- tolower(gsub("^Coord", "", coord_class))
      metadata[[ax_key]]$type_agnostic$axis <- list(
        position = NA,
        type = coord_name,
        aspect = if (inherits(p$coordinates, "CoordFixed")) p$coordinates$ratio else NA
      )
      metadata[[ax_key]]$type_agnostic$facet_wrap <- inherits(p$facet, "FacetWrap")
      title_text <- c(p$labels$title, p$labels$subtitle)
      title_text <- title_text[!sapply(title_text, is.null)]
      metadata[[ax_key]]$type_agnostic$title <- list(
        content = paste(title_text, collapse = "\n"),
        size = resolve_text_size(safe_theme_get(theme$plot.title, "size"), 11),
        style = safe_theme_get(theme$plot.title, "face")
      )
      metadata[[ax_key]]$type_agnostic$x_label <- list(
        content = p$labels$x,
        size = resolve_text_size(safe_theme_get(theme$axis.title.x, "size"), 11),
        style = safe_theme_get(theme$axis.title.x, "face")
      )
      metadata[[ax_key]]$type_agnostic$y_label <- list(
        content = p$labels$y,
        size = resolve_text_size(safe_theme_get(theme$axis.title.y, "size"), 11),
        style = safe_theme_get(theme$axis.title.y, "face")
      )
      metadata[[ax_key]]$type_agnostic$background_color <- safe_theme_get(theme$panel.background, "fill")
      panel_param <- pb$layout$panel_params[[1]]
      metadata[[ax_key]]$type_agnostic$colorbar <- has_colorbar(p)
      x_labels <- panel_param$x$breaks
      if (is.null(x_labels)) {
        if (!is.null(p$mapping$x) && !is.null(p$data)) {
          x_sym <- tryCatch(rlang::get_expr(p$mapping$x), error = function(e) NULL)
          x_name <- tryCatch(as.character(x_sym), error = function(e) NULL)

          if (!is.null(x_name) && x_name %in% names(p$data)) {
            x_labels <- unique(p$data[[x_name]])
          }
        }
      }
      x_pos <- seq_along(x_labels)
      metadata[[ax_key]]$type_agnostic$x_ticks <- mapply(function(lbl, idx) {
        list(text = as.character(lbl), position = c(idx, 0))
      }, x_labels, x_pos, SIMPLIFY = FALSE, USE.NAMES = FALSE)
      y_labels <- panel_param$y$breaks
      if (is.null(y_labels)) {
        if (!is.null(p$mapping$y) && !is.null(p$data)) {
          y_sym <- tryCatch(rlang::get_expr(p$mapping$y), error = function(e) NULL)
          y_name <- tryCatch(as.character(y_sym), error = function(e) NULL)

          if (!is.null(y_name) && y_name %in% names(p$data)) {
            y_labels <- unique(p$data[[y_name]])
          }
        }
      }
      y_pos <- seq_along(y_labels)

      metadata[[ax_key]]$type_agnostic$y_ticks <- mapply(function(lbl, idx) {
        list(text = as.character(lbl), position = c(0, idx))
      }, y_labels, y_pos, SIMPLIFY = FALSE, USE.NAMES = FALSE)
      legend_index <- which(sapply(gt$grobs, function(x) x$name) == "guide-box")
      if (length(legend_index) > 0) {
        metadata[[ax_key]]$type_agnostic$legend <- list(
          exist = TRUE,
          loc = NA,
          ncol = NA
        )
      } else {
        metadata[[ax_key]]$type_agnostic$legend <- list(exist = FALSE, loc = NA, ncol = NA )
      }
      is_boxed <- !inherits(p$theme$panel.border, "element_blank")
      metadata[[ax_key]]$type_agnostic$panel_box <- is_boxed
      grid_x <- !inherits(p$theme$panel.grid.major.x, "element_blank")
      grid_y <- !inherits(p$theme$panel.grid.major.y, "element_blank")
      metadata[[ax_key]]$type_agnostic$grid <- list(x = grid_x, y = grid_y)
      obj_result <- extract_objects_metadata(p)
      label_to_color <- get_label_to_color(p)
      metadata[[ax_key]]$object$patches <- obj_result$patches
      metadata[[ax_key]]$object$lines <- obj_result$lines
      metadata[[ax_key]]$object$collections <- obj_result$collections
      metadata[[ax_key]]$object$segments <- obj_result$segments
      metadata[[ax_key]]$type_agnostic$label_to_color <- label_to_color
      metadata[[ax_key]]$data_table <- p$data
      annotations <- get_annotations(obj_result$texts)
      metadata[[ax_key]]$type_agnostic$annotation <- annotations
    }
    
    return(metadata)

  }, error = function(e) {
    print(e)
    return(list(
      execute = list(
        source_file = label,
        language = "r",
        executable = FALSE,
        error = list(
          type = class(e)[1],
          message = conditionMessage(e),
          call = deparse(conditionCall(e))
        )
      )
    ))
  })

  return(result)
}

sanitize_for_json <- function(x) {
  if (inherits(x, "expression") || inherits(x, "unit") ||
      inherits(x, "rel") || inherits(x, "element_text")) {
    return(as.character(x))
  } else if (is.function(x) || is.environment(x)) {
    return(NA)
  } else if (is.list(x)) {
    return(lapply(x, sanitize_for_json))
  } else {
    return(x)
  }
}

suppressPackageStartupMessages({
  library(optparse)
  library(R.utils)
})
opt_list <- list(
  make_option("--head_dir"),
  make_option("--suffix")
)
opt <- parse_args(OptionParser(option_list = opt_list))

script_files <- list.files(
  opt$head_dir,
  pattern     = "\\.R$",
  recursive   = TRUE,
  full.names  = TRUE
)

for (script_path in script_files) {
  code_str <- paste(readLines(script_path, warn = FALSE), collapse = "\n")
  if (!grepl("ggplot2\\s*::\\s*ggplot\\s*\\(|\\bggplot\\s*\\(", code_str)) {
    warning("ggplot2 code not detected; skipping base R metadata extraction.")
    next
  }

  cat("Processing:", script_path, "\n")

  metadata <- NULL
  tryCatch({
    metadata <- withTimeout(
      extract_metadata_from_r_runtime(code_str, script_path),
      timeout = 20,
      onTimeout = "silent"
    )
  }, error = function(e) {
    warning(sprintf("Error while processing %s: %s", script_path, conditionMessage(e)))
  })

  if (is.null(metadata)) {
    warning(sprintf("Skipping %s due to NULL metadata or timeout.", script_path))
    next
  }
  
  tryCatch({
    metadata <- sanitize_for_json(metadata)
    json_str <- jsonlite::toJSON(metadata, auto_unbox = TRUE, pretty = TRUE, null = "null", force = TRUE)
    save_path <- sub("\\.R$", "_object.json", script_path)
    write(json_str, file = save_path)
    cat("Saved metadata to:", save_path, "\n")
  }, error = function(e) {
    warning(sprintf("Skipping %s due to error: %s", script_path, conditionMessage(e)))
  })
}