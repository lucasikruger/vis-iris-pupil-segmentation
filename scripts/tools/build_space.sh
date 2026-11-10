#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
space_output="${1:-$project_root/dist-static-space}"
if [[ -e "$space_output" ]]; then
  echo "La salida ya existe: $space_output. Elegí otro directorio." >&2
  exit 1
fi
(cd "$project_root/demos/web" && npm ci && npm run build)
mkdir -p "$space_output"
cp -R "$project_root/demos/web/dist/." "$space_output/"
cp "$project_root/demos/web/space/README.md" "$space_output/README.md"
cp "$project_root/LICENSE" "$space_output/LICENSE"
cp "$project_root/weights/LICENSE" "$space_output/MODEL_LICENSE"
cp "$project_root/data/LICENSE" "$space_output/DATA_LICENSE"
cp "$project_root/docs/THIRD_PARTY.md" "$space_output/THIRD_PARTY.md"
cp -R "$project_root/docs/licenses" "$space_output/licenses"
sed -i 's#(../data/DATA.md)#(https://github.com/lucasikruger/vis-iris-pupil-segmentation/blob/main/data/DATA.md)#g; s#`docs/licenses/#`licenses/#g' "$space_output/THIRD_PARTY.md"
printf 'Space estático: %s\n' "$space_output"
