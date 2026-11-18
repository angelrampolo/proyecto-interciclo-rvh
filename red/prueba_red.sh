#!/usr/bin/env bash
# Prueba rápida de usabilidad: 30 min de viajes aleatorios (2 veh/s), 1 h simulada.
# Uso: bash red/prueba_red.sh [semilla]
set -euo pipefail
cd "$(dirname "$0")/.."
SEED=${1:-42}
docker run --rm --ulimit core=0 -u "$(id -u):$(id -g)" -v "$PWD":/w -w /w --tmpfs /t:mode=1777 van3twin:jammy bash -c "
python3 /usr/share/sumo/tools/randomTrips.py -n red/cuenca_urbana.net.xml -o /t/trips.xml -e 1800 -p 0.5 --fringe-factor 5 --min-distance 1000 --seed $SEED --validate -r /t/r.rou.xml >/dev/null 2>&1
sumo -n red/cuenca_urbana.net.xml -r /t/r.rou.xml -e 3600 --no-step-log --duration-log.statistics --time-to-teleport 300 2>&1 | grep -vE '^Warning' | tail -18"
