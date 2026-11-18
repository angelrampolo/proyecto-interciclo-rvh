#!/usr/bin/env bash
# Construye y poda la red SUMO de Cuenca urbana (15 parroquias urbanas).
# Mismo flujo de las prácticas (results/cuenca_osm): netconvert estilo osmWebWizard
# -> prune_deadends -> netconvert de limpieza (no-turnarounds, isolated).
# Solo autos (passenger): se descartan las vías de servicio (parqueaderos, accesos
# privados). red/uniones.nod.xml une intersecciones que quedaban partidas en varios
# nodos y provocaban gridlock (ver DOCUMENTACION.md).
# Uso: bash red/construir_red.sh      (desde "Proyecto Interciclo")
set -euo pipefail
cd "$(dirname "$0")/.."
IMG=van3twin:jammy
run() { docker run --rm --ulimit core=0 -u "$(id -u):$(id -g)" -e SUMO_HOME=/usr/share/sumo -v "$PWD":/w -w /w "$IMG" "$@"; }
log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

log "Fase 1/3: netconvert desde OSM (osm/cuenca_urbana.osm)"
run netconvert --type-files red/tipos_cuenca.typ.xml --osm-files osm/cuenca_urbana.osm --node-files red/uniones.nod.xml -o red/cuenca_bruta.net.xml \
  --proj.utm --geometry.remove --roundabouts.guess --ramps.guess --junctions.join \
  --tls.guess-signals --tls.discard-simple --tls.join --tls.default-type actuated \
  --remove-edges.by-type highway.service,highway.living_street \
  --keep-edges.by-vclass passenger --keep-edges.components 1 --remove-edges.isolated \
  --osm.oneway-spread-right false --output.street-names \
  --junctions.corner-detail 5 --junctions.limit-turn-speed 5.5 \
  --aggregate-warnings 5 2>&1 | grep -Ev '^Warning' | tail -5

log "Fase 2/3: poda iterativa de calles sin salida (prune_deadends, sin tocar el tranvía)"
run python3 red/prune_deadends.py red/cuenca_bruta.net.xml red/prune.txt

log "Fase 3/3: netconvert de limpieza (remove-edges, no-turnarounds, isolated)"
run netconvert -s red/cuenca_bruta.net.xml -o red/cuenca_urbana.net.xml \
  --remove-edges.input-file red/prune.txt --remove-edges.isolated \
  --keep-edges.components 1 --no-turnarounds \
  --junctions.corner-detail 5 --junctions.limit-turn-speed 5.5 \
  --offset.disable-normalization --aggregate-warnings 5 2>&1 | grep -Ev '^Warning' | tail -5

rm -f red/cuenca_bruta.net.xml
log "Listo: red/cuenca_urbana.net.xml"
