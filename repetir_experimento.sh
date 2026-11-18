#!/usr/bin/env bash
# Repite el experimento completo del Proyecto Interciclo RVH (Cuenca, SUMO + V2X).
#
#   ./repetir_experimento.sh                 # entrena y evalua todo desde el dataset (dataset/ debe existir)
#   ./repetir_experimento.sh --desde-cero    # ademas regenera la red y el dataset sintetico en SUMO (~13 h)
#   ./repetir_experimento.sh --solo-figuras  # solo regenera tablas, figuras, tablero y pruebas desde resultados/ (~2 min)
#
# Variables: PY (interprete de Python, por defecto python3), PARALELO (simulaciones SUMO a la vez, por defecto 4),
#            RVH_DATOS (carpeta de datos; por defecto dataset/, ver FORMATO_DATOS.md para usar datos reales).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
PARALELO=${PARALELO:-4}
MODO=${1:-normal}
paso() { printf '\n==== %s  (%s)\n' "$1" "$(date +%H:%M:%S)"; }

if [ "$MODO" = "--desde-cero" ]; then
  docker image inspect van3twin:jammy >/dev/null || { echo "Falta la imagen Docker van3twin:jammy (SUMO 1.12)"; exit 1; }
  paso "1. Red vial desde OpenStreetMap (netconvert + poda)";      bash red/construir_red.sh
  paso "2. Informacion de la red para Python"
  docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/w -w /w van3twin:jammy python3 escenario/exportar_red.py red/cuenca_urbana.net.xml escenario/red_info.json
  paso "3. Punto de saturacion V0 (3 semillas)"
  $PY etapa2/barrido_saturacion.py --tasas 4000 8000 12000 14000 15000 16000 17000 18000 20000 24000 --semillas 101 202 303 --paralelo "$PARALELO"
  paso "4. Dataset sintetico de 46 dias (semilla = 1000 + dia)";   $PY etapa2/generar_dataset.py --todos --paralelo "$PARALELO"
  paso "5. 20 dias extra con incidentes para el detector";        $PY v2/generar_dias_incidentes.py --paralelo "$PARALELO"
  paso "6. Validacion del dataset";                               $PY etapa2/validar_dataset.py
fi

if [ "$MODO" != "--solo-figuras" ]; then
  paso "Formato de los datos";                                    $PY v2/validar_datos.py
  paso "Validacion del sintetico frente a cifras reales de Cuenca"; (cd v2 && $PY validar_sintetico.py)
  paso "7. Prediccion GCN-GRU con sensores y con V2X 50 %"
  $PY v2/entrenar_prediccion.py --modo gt
  $PY v2/entrenar_prediccion.py --modo obs --epocas 8
  paso "8. Detector de incidentes (umbral sin test, comparacion con el azar)"
  $PY v2/entrenar_aid.py --modo gt
  $PY v2/entrenar_aid.py --modo obs
  paso "9. Bucle cerrado SUMO + TraCI + DENM (con y sin IA)";      $PY v2/sil_sumo.py --inc inc119_3 --aid gt
fi

paso "10. Tablero";                                               $PY v2/construir_dashboard.py
paso "11. Figuras del informe";                                   $PY v2/informe/figuras_informe.py
paso "12. Banco de pruebas (actualiza el conteo del informe)";    $PY -m pytest v2/pruebas -q -p no:cacheprovider || true
if command -v tectonic >/dev/null; then
  paso "13. Informe y diapositivas en PDF"
  (cd v2/informe && for d in informe diapositivas; do tectonic "$d.tex"; done)
fi
echo; echo "Listo. Resultados en v2/resultados/, figuras en v2/informe/figuras/, tablero: python3 v2/servir_dashboard.py"
