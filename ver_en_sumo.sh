#!/usr/bin/env bash
# Ver y repetir en SUMO el aviso V2X (DENM) ante el cierre de la Av. de las Americas (incidente inc119_3, dia 119).
#
#   ./ver_en_sumo.sh              abre sumo-gui con la corrida CON IA: naranja = auto que iba al cierre,
#                                 azul = auto que recibio el DENM y recalculo su ruta; el cierre esta marcado en rojo
#   ./ver_en_sumo.sh base         abre sumo-gui con la corrida SIN IA (los mismos autos, nadie avisa)
#   ./ver_en_sumo.sh replicar     repite las dos corridas sin ventana y regenera los resultados y el tablero (~30 min)
#
# La primera vez prepara las rutas del dia con duarouter (~15 min); despues se reutilizan.
# El cierre empieza a las 14:10; la simulacion arranca a las 12:40 para llenar la red (en sumo-gui se puede
# acelerar con el control "Delay" y pausar con el boton de pausa).
# Requisitos: Docker con la imagen van3twin:jammy (SUMO 1.12), Python con requirements.txt y traci,
# y una sesion grafica (X11 o Wayland con XWayland).
set -euo pipefail
cd "$(dirname "$0")/v2"
if [ -z "${PY:-}" ]; then
  if [ -x "$HOME/venv_cpu/bin/python" ]; then PY="$HOME/venv_cpu/bin/python"; else PY=python3; fi
fi
export SIL_TRABAJO="${SIL_TRABAJO:-/tmp/rvh_sil}"
docker image inspect van3twin:jammy >/dev/null || { echo "Falta la imagen Docker van3twin:jammy (SUMO 1.12)"; exit 1; }

case "${1:-ia}" in
  replicar)
    echo "Repitiendo las dos corridas (sin IA y con IA) en SUMO..."
    $PY sil_sumo.py --inc inc119_3 --aid gt
    $PY construir_dashboard.py
    echo "Listo: resultados/sil_inc119_3_gt.json, resultados/sil_recorridos_inc119_3_gt.json y el tablero actualizados."
    ;;
  ia|base)
    command -v xhost >/dev/null && xhost +SI:localuser:"$(id -un)" >/dev/null 2>&1 || true
    echo "Abriendo sumo-gui (corrida: ${1:-ia}). No se modifica ningun resultado."
    SIL_GUI=1 SIL_VER="${1:-ia}" $PY sil_sumo.py --inc inc119_3 --aid gt
    ;;
  *)
    sed -n '2,14p' "$0"; exit 1 ;;
esac
