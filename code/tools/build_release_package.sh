#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_TAR_GZ" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output_path="$1"
if [[ "$output_path" != /* ]]; then
  output_path="$PWD/$output_path"
fi
mkdir -p "$(dirname "$output_path")"

staging_root="$(mktemp -d)"
trap 'rm -rf "$staging_root"' EXIT
package_root="$staging_root/easylens"
mkdir -p "$package_root"

rsync -a \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude '*.ckpt' \
  --exclude '*.safetensors' \
  --exclude '*.bin' \
  --exclude '*.onnx' \
  --exclude '*.nii' \
  --exclude '*.nii.gz' \
  --exclude 'benchmarks/*/images/' \
  --exclude 'assets/figure_*.png' \
  --exclude 'dist/' \
  --exclude '*.log' \
  --exclude '.DS_Store' \
  "$repo_root/" "$package_root/"

python3 "$package_root/code/tools/validate_release_package.py" --root "$package_root"
COPYFILE_DISABLE=1 tar --no-xattrs --no-acls --no-fflags -C "$staging_root" -czf "$output_path" easylens
shasum -a 256 "$output_path"
