#!/bin/bash

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate charluma

# ---------- parameter setting ----------

HEAD_DIR="your_data_dir"

ORIG_LANG="python"
TRANS_LANG=("r" "latex")

# ORIG_LANG="r"
# TRANS_LANG=("python" "latex")

# ORIG_LANG="latex"
# TRANS_LANG=("python" "r")

# ---------- helping functions ----------

START_TIME=$(date +%s)
report_time() {
    local label="$1"
    local now=$(date +%s)
    local elapsed=$(( now - START_TIME ))
    echo "⏱  [${label}]  ${elapsed}s since start"
}

run_one() {
  local file="$1" sfx="$2" limit="${3:-60s}"
  echo "→ Running: $file (timeout: $limit)"

  case "$sfx" in
    .py)
      timeout --preserve-status --kill-after=5s "$limit" python "$file"
      ;;
    .R|.r)
      timeout --preserve-status --kill-after=5s "$limit" ./miniconda3/envs/r_studio/bin/R < "$file" --save --quiet --no-echo
      ;;
    .tex)
      ( cd "$(dirname "$file")" \
        && timeout --preserve-status --kill-after=5s "$limit" \
             pdflatex -interaction=nonstopmode "$(basename "$file")" >/dev/null )
      ;;
    *)
      echo "Skip (unknown suffix $sfx): $file"
      return 0
      ;;
  esac

  local status=$?
  case "$status" in
    0)   echo "✅ Done: $file" ;;
    124) echo "⏱️ Timed out after $limit: $file" ;;
    137) echo "💥 Killed after grace period: $file" ;;
    *)   echo "❌ Exit code $status: $file" ;;
  esac
  return "$status"
}

lang_suffix() {
  local l="${1,,}"
  if [[ "$l" == "python" || "$l" == "py" ]]; then
    echo ".py"
  elif [[ "$l" == "r" || "$l" == "ggplot" ]]; then
    echo ".R"
  elif [[ "$l" == "latex" || "$l" == "tex" ]]; then
    echo ".tex"
  else
    echo ""
    return 1
  fi
}

if [[ -z "${SUFFIX:-}" ]]; then
  SUFFIX="$(lang_suffix "$ORIG_LANG")" || { echo "Unsupported ORIG_LANG: $ORIG_LANG"; exit 1; }
else
  [[ "$SUFFIX" == .* ]] || SUFFIX=".$SUFFIX"
fi

for tl in "${TRANS_LANG[@]}"; do
  s="$(lang_suffix "$tl")" || { echo "Unsupported trans lang: $tl"; exit 1; }
  TRANS_SUFFIX+=("$s")
done


# ---------- metadata extraction ----------

report_time "Metadata Extraction start"

if [[ "${ORIG_LANG,,}" == "python" ]]; then
  python ./main/extractor/extract_metadata_python.py \
    --head_dir "$HEAD_DIR" \
    --suffix "$SUFFIX"
elif [[ "${ORIG_LANG,,}" == "r" ]]; then
  ./miniconda3/envs/r_studio/bin/Rscript ./main/extractor/extract_metadata_r.R \
    --head_dir "$HEAD_DIR" \
    --suffix "$SUFFIX"
elif [[ "${ORIG_LANG,,}" == "latex" ]]; then
  python ./main/extractor/extract_metadata_latex.py \
    --head_dir "$HEAD_DIR" \
    --suffix "$SUFFIX"
fi

# ---------- template filling ----------

report_time "Template Filling start"

python ./main/worker/template_filling.py \
    --head_dir "$HEAD_DIR" \
    --trans_langs "${TRANS_LANG[@]}"

# ---------- execute scrips ----------

report_time "Execute Scripts start"

for i in "${!TRANS_LANG[@]}"; do
  lang="${TRANS_LANG[$i]}"
  sfx="${TRANS_SUFFIX[$i]}"
  while IFS= read -r -d '' f; do
    run_one "$f" "$sfx" || echo "⚠️  Failed: $f"
  done < <(find "$HEAD_DIR" -type f -iname "*_${lang}${sfx}" -print0)
done

report_time "All done"