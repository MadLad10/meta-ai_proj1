#!/usr/bin/env bash
# Build main.pdf: regenerate numbers/tables/figure from the analysis outputs, then compile with TeX Live in Docker.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
"$PY" ../analysis/crops.py > /dev/null
"$PY" ../analysis/make_tex.py
"$PY" ../analysis/make_crop_fig.py
"$PY" ../analysis/make_vlm_fig.py
"$PY" ../analysis/make_examples.py   # DUO images are cached after the first run
docker run --rm --memory 1g -u "$(id -u):$(id -g)" -v "$PWD:/work" -w /work auv2-latex:3 \
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
pdfinfo main.pdf | grep Pages
