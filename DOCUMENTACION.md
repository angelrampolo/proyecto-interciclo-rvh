# Proyecto Interciclo RVH — Escenario de toda Cuenca y dataset (Etapas 1 y 2)

**Asignatura:** Redes Vehiculares Heterogéneas. **Objetivo:** predecir el estado del tráfico de Cuenca con 1 h de anticipación (CNN-LSTM/GNN-LSTM) y detectar incidentes (AID) usando solo datos de vehículos conectados (V2X).

Este documento describe cómo se construyó la red de **toda Cuenca urbana**, cómo se generó el dataset y cómo reproducirlo. Todo corre con el contenedor `van3twin:jammy` (SUMO 1.12) y Python del anfitrión (`~/venv_cpu`, con pandas, pyarrow, matplotlib y torch).

## Decisiones de diseño

| Decisión | Motivo |
|---|---|
| Área = las **15 parroquias urbanas** de Cuenca (límites oficiales de OSM) | El profesor pidió toda Cuenca. Las parroquias rurales del cantón (3.665 km²) no aportan a un modelo urbano y harían inviable la simulación |
| **Solo autos** (`passenger`), 50 % conectados (V2X) | Pedido explícito. Además, los cruces del tranvía hacían abortar SUMO 1.12 (`getBackPositionOnLane` assertion) |
| **Sin rerouting** | Recomendación del profesor: rutas fijas, sin `device.rerouting` |
| Rutas con **duarouter** | Recomendación del profesor. Rutas de menor tiempo en red libre con `--weights.random-factor 1.5` para repartir el tráfico entre vías paralelas |
| Demanda pico = **punto de saturación V0** | Recomendación del profesor: la hora pico se simula en la saturación real de la red |
| Días independientes **en paralelo** (9 contenedores) | Recomendación del profesor. SUMO no guarda memoria entre días, así que correr días separados equivale a semanas continuas |
| Humano y V2X con el **mismo modelo de conducción** | Conectado ≠ automatizado. La vista V2X es una muestra aleatoria del 50 % del tráfico, que es lo que verá el modelo en la Etapa 4 |
| Incidentes sin TraCI (autos detenidos con `<stop>`) | Corren a la velocidad de SUMO nativo. TraCI queda para el bucle en vivo (Etapa 4) |
| `--ignore-junction-blocker 30` | Sin él, SUMO forma bloqueos circulares en los cruces (gridlock) que no se deshacen hasta la medianoche, incluso en días de demanda baja. Ver § 2.1 |

## 1. Red vial (Etapa 1 rehecha para toda Cuenca)

**Archivos:** `osm/cuenca_urbana.osm`, `red/construir_red.sh`, `red/prune_deadends.py`, `red/tipos_cuenca.typ.xml`, `red/cuenca_urbana.net.xml`.

1. **Descarga OSM (Overpass):** vías `motorway…residential` y sus enlaces, dentro de las 15 parroquias urbanas. Overpass no crea áreas para relaciones sin nombre, así que Hermano Miguel y Machángara (sin nombre en OSM) se filtran con su polígono (`way(poly:…)`).
2. **netconvert** con las mismas opciones del asistente OSM usadas en las prácticas (`results/cuenca_osm`): `--geometry.remove --roundabouts.guess --ramps.guess --junctions.join --tls.guess-signals --tls.discard-simple --tls.join`, más `--tls.default-type actuated`, `--keep-edges.by-vclass passenger` y la eliminación de `highway.service` y `living_street` (parqueaderos y accesos privados).
3. **Velocidades realistas** (`red/tipos_cuenca.typ.xml`): si la vía tiene `maxspeed` en OSM se respeta (4.091 vías lo tienen). Si no, se aplica la Ley de Tránsito de Ecuador: 50 km/h en zona urbana, 70–80 km/h en vías rápidas y 40 km/h en enlaces. Sin esto, SUMO asigna 100 km/h a las avenidas primarias.
4. **Poda** (`prune_deadends.py`, igual que en las prácticas): elimina de forma iterativa las calles sin salida (4.547 aristas en 6 pasadas). Además elimina las aristas que quedan fuera de la mayor componente fuertemente conexa (69), para que cualquier origen llegue a cualquier destino.
5. **netconvert de limpieza:** `--remove-edges.input-file`, `--no-turnarounds`, `--remove-edges.isolated`.
6. **Uniones manuales** (`red/uniones.nod.xml`, pasado a la fase 1 con `--node-files`): une la intersección de Av. de las Américas con Av. Miraflores, Antonio Farfán y Mariano Cueva, que netconvert dejó partida en dos cruces (ver § 2.2).

| Indicador | Valor |
|---|---|
| Extensión | 20,0 × 12,2 km (lon −79,080 a −78,888; lat −2,939 a −2,825) |
| Aristas / nodos | 16.274 / ~7.400 |
| Longitud de vías / km-carril | 1.350 km / 1.675 km |
| Semáforos (actuados) | 360 |
| Sentido único / doble sentido | 4.839 / 11.439 aristas |
| Conectividad | 100 % de aristas en una sola componente fuertemente conexa |

**Mapa:** `escenario/mapa_cuenca_urbana.pdf`, con 4 láminas: jerarquía vial y semáforos, límites de velocidad, sentidos de circulación con zoom al Centro Histórico, y malla con corredores. Se regenera con `escenario/mapa_pdf.py`.

**Malla:** celdas de 200 m, 100 × 61, con **1.694 celdas activas**. Cada arista se asigna a la celda de su punto medio.

**Corredores AID:**

| Corredor | Vías | Longitud |
|---|---|---|
| C1_Americas | Av. de las Américas | 24,3 km |
| C2_Circunvalacion_Sur | Circunvalación Sur | 13,2 km |
| C3_RemigioCrespo_Solano | Av. Remigio Crespo Toral y Av. Fray Vicente Solano | 6,1 km |

Las longitudes cuentan ambos sentidos.

## 2. Punto de saturación V0 (recomendación del profesor)

**Script:** `etapa2/barrido_saturacion.py`. **Resultados:** `etapa2/resultados/barrido_saturacion*.csv` y `barrido_saturacion.png`.

Para cada demanda R se insertan viajes de solo autos a tasa constante durante 75 min (15 de llenado y 60 de medición), con rutas de duarouter y sin rerouting. Cada demanda se corrió con 3 semillas (101, 202, 303) y se promedia.

**Criterio:** V0 es la mayor demanda (promedio de semillas) que cumple a la vez:
- eficiencia (throughput / demanda) ≥ 0,95,
- cola de inserción < 2 % de R,
- teletransportes < 1 % de R,
- velocidad media ≥ 75 % de la de baja demanda.

**Resultado: V0 = 17.000 veh/h**, con la red final (intersección Américas–Miraflores unida, § 2.2) y `--ignore-junction-blocker 30` (§ 2.1).

| Demanda (veh/h) | Throughput (veh/h) | Eficiencia | Velocidad media (km/h) | Teletransportes en 1 h |
|---|---|---|---|---|
| 4.000 | 3.992 | 1,00 | 40,2 | 0 |
| 8.000 | 7.961 | 1,00 | 39,5 | 0 |
| 12.000 | 11.985 | 1,00 | 38,7 | 0 |
| 14.000 | 13.831 | 0,99 | 38,0 | 0 |
| 15.000 | 14.815 | 0,99 | 37,2 | 3 |
| 16.000 | 15.603 | 0,98 | 35,5 | 2 |
| **17.000** | **16.295** | **0,96** | **33,7** | **27** |
| 18.000 | 16.694 | 0,93 | 30,8 | 62 |
| 20.000 | 17.550 | 0,88 | 26,8 | 341 |
| 24.000 | 17.956 | 0,75 | 20,4 | 1.303 |

Por encima de 17.000 veh/h la red deja de atender la demanda. El throughput se estanca en ~17.500–18.000 veh/h (la capacidad de la red), la velocidad cae y los teletransportes crecen rápido. El viaje medio es de 5,8 km y ~9 min en red libre. Historial de V0: 15.000 con la red original sin la opción, 16.000 con la opción y 17.000 con la intersección corregida.

### 2.1 Primera corrección de bloqueos: `--ignore-junction-blocker`

En la primera generación (V0 = 15.000, sin la opción) dos días quedaron atascados de forma irreal:

| Día | Tipo | Velocidad media | Teletransportes | Qué pasó |
|---|---|---|---|---|
| 4 | laborable ×1,04, sin incidentes | 26 km/h | 19.452 | Bloqueo desde las 17:40 en Av. de las Américas y Abelardo J. Andrade; no se recuperó hasta la medianoche |
| 13 | sábado ×0,93 (pico ≈ 10.400 veh/h, muy por debajo de V0) | 22 km/h | 14.685 | Bloqueo desde las 10:15 en Gil Ramírez Dávalos y Sebastián de Benalcázar |

Un sábado con demanda baja no puede colapsar por exceso de autos: era un **bloqueo circular** de SUMO, en el que cada auto espera a otro que está dentro del cruce y la cola da la vuelta a la manzana. Ningún conductor real se queda 5 minutos parado dentro de una intersección. Los motivos de los teletransportes del día 13 fueron *jam* y *yield* en el nodo 413678740, que estaba vacío por delante.

**Prueba** (`etapa2/prueba_bloqueo.py`): se reprodujeron los dos días con la misma semilla.

| | Sin opción | Con `--ignore-junction-blocker 30` |
|---|---|---|
| Día 13 a las 12:00 | 3.342 autos en red, 1.732 detenidos, 1.368 teletransportes | 1.732 en red, 94 detenidos, 1 teletransporte |
| Día 4 a las 19:00 | 3.096 en red, 1.071 detenidos (en aumento) | 2.507 en red, 290 detenidos, 34 teletransportes |

Con la opción, un auto que lleva más de 30 s detenido dentro de un cruce deja de bloquear a los que cruzan, como hacen los conductores reales al abrirse paso. Se dejó fija en `comun.correr_sumo`. Como cambia la capacidad de los cruces, **se repitió el barrido** (tabla anterior; el resultado viejo está en `etapa2/resultados/sin_ijb/`) y se regeneró el dataset.

### 2.2 Segunda corrección: intersección partida en Américas y Miraflores

Con la opción anterior, la segunda generación seguía teniendo días que colapsaban sin recuperarse: los días 1, 2, 5, 11, 12 y 15 de los primeros 19, con más de 40.000 autos detenidos y velocidad media cercana a 0. El día 12 colapsó con demanda ×0,95, mientras que el día 4, con ×1,04, salió bien. Eso descartaba el exceso de demanda como causa.

**Diagnóstico** (edgeData de 5 min): en todos los días el colapso empieza entre 7:30 y 8:00 en la **Av. de las Américas, entre Av. Miraflores y Mariano Cueva**. Aguas abajo del cruce, la vía estaba vacía, a 56 km/h. Aguas arriba, la ocupación era de 65 % a 0 km/h. OSM dibuja la Américas como doble calzada, y netconvert la dejó como **dos cruces separados ~60 m** (uno con semáforo y otro de prioridad), con tramos de 13–35 m entre ellos. Ahí solo caben 2 o 3 autos por carril. Los giros a la izquierda de cada calzada esperaban a la cola de la otra, y esa cola esperaba a los primeros: un bloqueo mutuo que ninguna opción de SUMO deshace. Desde ahí la cola se extendía por toda la Américas y el resto de la ciudad.

**Corrección:** `red/uniones.nod.xml` une los 7 nodos OSM de la intersección en un solo cruce con semáforo, como funciona en la realidad. La red pasa a 16.274 aristas y 360 semáforos; las celdas y los corredores no cambian.

**Prueba** con las mismas semillas, hasta las 12:00 (`etapa2/resultados/prueba_union_dias_2_5_12.txt`):

| Día (factor) | Antes | Después: peor momento | Después: 11:00 |
|---|---|---|---|
| 2 (×1,12) | colapso total | 9:00, 26 km/h, 1.099 detenidos | 38 km/h, 165 detenidos, 264 teletransportes en total |
| 5 (×1,10) | colapso total | 9:00, 24 km/h, 1.326 detenidos | 36 km/h, 172 detenidos, 388 teletransportes |
| 12 (×0,95) | colapso total | 8:00, 36 km/h, 280 detenidos | 40 km/h, 84 detenidos, 9 teletransportes |

Ahora el pico de la mañana se congestiona y se despeja hacia las 10:00, como en la realidad. Otros puntos se cargan en la hora pico (Gran Colombia, Paseo Río Machángara, Barrial Blanco y Héroes de Verdeloma, que son entradas a la ciudad), pero se recuperan solos. Con la red corregida **se repitió el barrido de V0** y **se regeneró todo el dataset**. Los resultados de las versiones anteriores quedan en `etapa2/resultados/sin_ijb/` y `etapa2/resultados/red_v1_ijb/`.

## 3. Demanda del dataset

- **Pico = V0 × factor del día.** El factor se sortea entre 0,88 y 1,07. Con lluvia se multiplica por 1,04, de modo que el pico máximo queda en ~×1,11. Unos días quedan bajo la saturación y otros algo por encima, así el modelo ve tanto flujo estable como congestión.
  - **Por qué no más:** con rutas fijas (sin rerouting), los conductores no pueden evitar un atasco. Si la demanda supera a V0 en más de ~12 % durante el pico, la congestión ya no se disuelve: con V0 = 17.000, el día de estrés a ×1,25 acumuló 19.000 autos detenidos a las 12:00. En cambio, el día 2 a ×1,12 se congestionó fuerte a las 9:00 (17 km/h) y se recuperó hacia las 11:00. Pruebas en `etapa2/resultados/prueba_v0_17000_dia*.txt`.
  - **Vigilante de colapso:** si en algún día la red acumula más de 10.000 autos a la vez (un pico normal tiene 3.000–5.000), `comun.correr_sumo` corta la simulación y `generar_dataset` repite el día con 5 % menos de demanda, hasta 3 intentos. `factor_efectivo` y `reintentos` quedan en `dias.parquet`.
- **Perfiles horarios** (fracción de V0) en `etapa2/comun.py → PERFILES`:
  - laborable: picos 07–08 (1,00), 12–14 (0,85–0,92) y 18–19 (0,98), valle nocturno 0,03;
  - sábado: pico 0,80 al mediodía;
  - domingo: pico 0,62.
  
  La tasa es constante dentro de cada hora (con transiciones suavizadas de 20 min), de modo que la hora pico promedia exactamente V0 × factor. Un día laborable con factor 1 tiene ~200.000 viajes.
- **Viajes** (`generar_viajes`): proceso de Poisson no homogéneo. Origen y destino se ponderan por longitud × carriles, como en `randomTrips`; las aristas del borde de la red pesan ×3 para representar el tráfico que entra y sale de la ciudad. Hay un decaimiento gravitacional con la distancia (P = e^−(d−1 km)/5 km) y una distancia mínima de 1 km.
- **Rutas:** `duarouter --routing-algorithm astar --weights.random-factor 1.5`. La perturbación aleatoria reparte el tráfico entre vías paralelas, como hacen los conductores reales.
- **Tipos** (`escribir_vtypes`): `auto_humano` y `auto_v2x` son idénticos (Krauss, σ = 0,5, τ = 1,0, speedFactor ~ N(1; 0,1)); solo cambia la etiqueta. Con **lluvia** (20 % de los días): speedFactor medio 0,88 y demanda +4 %.
- **Simulación:** `--time-to-teleport 300 --max-depart-delay 900 --ignore-junction-blocker 30`, sin rerouting. Cada día va de 23:00 del día anterior a 24:00; la primera hora es de calentamiento y se descarta.

## 4. Incidentes

- Entre 3 y 6 incidentes en el 70 % de los días, entre las 05:00 y las 22:00, con duración de 10 a 25 min.
- **Ubicación:**
  - 70 % en los corredores (primero se elige el corredor con igual probabilidad entre C1, C2 y C3) y 30 % en otras vías de ≥ 50 km/h; siempre en tramos de ≥ 120 m.
  - Dentro de cada grupo, la probabilidad es proporcional al número de rutas que pasan por la arista (solo entre las que superan la mediana de uso): los accidentes ocurren donde hay tráfico.
- **Severidad:** en vías de 1 carril el bloqueo es total. En vías de varios carriles, 75 % bloquean 1 carril, 15 % bloquean 2 (solo en vías de 3 o más) y 10 % son cierre total.
  - *Por qué:* en una prueba con 30 % de cierres totales, cerrar los 3 carriles del tramo más usado de la Av. de las Américas a las 12:18, durante solo 24 min, trabó la ciudad ~5 h. Sin rerouting los autos no se desvían. Lo realista es que la mayoría de los accidentes bloqueen 1 o 2 carriles.
- **Implementación:** un auto `incidente` detenido (`<stop>` en la vía) por cada carril bloqueado. No se usa TraCI.
- **Verdad de terreno:** el inicio y el fin reales se toman de `--stop-output`, no del plan.

## 5. Calendario y división

`dataset/calendario.json`, generado con semilla 2026.

| Split | Días | Contenido |
|---|---|---|
| train | 1–28 (semanas 1–4) | 20 laborables, 4 sábados, 4 domingos |
| val | 29–35 (semana 5) | 5 laborables, 1 sábado, 1 domingo |
| test | 36–42 (semana 6) | 5 laborables, 1 sábado, 1 domingo |
| test_estres | 43–46 | demanda nominal ×1,12 (la máxima que la red recupera) y ×1,07 con lluvia; penetración V2X del 30 % y del 70 %. Los días 43 y 44 se repitieron una vez y quedaron en ×1,064 y ×1,057 (ver §9) |

La división se hace **por días completos**, nunca por muestras al azar, para que no se filtren datos de prueba al entrenamiento.


## 6. Estructura del dataset (`dataset/`)

Todas las tablas están en Parquet (zstd), con un archivo por día dentro de cada carpeta. Se leen completas con `pd.read_parquet('dataset/celdas_5min')`, o por días con `filters=[('dia_id','in',[1,2])]`. El tiempo `t` se mide en segundos desde las 00:00 e indica el **inicio** del intervalo.

| Tabla | Granularidad | Uso |
|---|---|---|
| `celdas_5min/` | día × 288 intervalos × 1.694 celdas | Tensores de entrada y salida del CNN-LSTM / GNN-LSTM (predicción a 1 h) |
| `corredor_1min/` | día × 1.440 minutos × ~311 aristas de corredor (por sentido) | Clasificador AID |
| `red_5min/` | día × 288 | Estado global de la red (autos circulando, cola, velocidad, llegadas, teletransportes) |
| `viajes/` | un registro por viaje | Métricas de evaluación de la Etapa 5 (tiempo de viaje, pérdida de tiempo) |
| `dias.parquet` | un registro por día | Parámetros del día, split y estadísticas |
| `incidentes.parquet` | un registro por incidente | Verdad de terreno de los incidentes |
| `celdas.parquet`, `aristas.parquet` | estático | Malla (fila, col, v_ff, km-carril, corredor) y aristas (celda, corredor, carriles, velocidad) |

**Dos vistas de los mismos datos:**
- `gt_*`: verdad de terreno, con **todos** los autos.
- `obs_*`: observada V2X, solo los autos **conectados**. Es lo único que verá el modelo en la Etapa 4.

Las vistas se miden con `edgeData` filtrado por `vTypes` (cada 5 min) y con detectores E1 filtrados por `vTypes` (cada 1 min). Los vehículos `incidente` no se cuentan en ninguna.

**Columnas de `celdas_5min`:**

| Columna | Unidad | Significado |
|---|---|---|
| `{gt,obs}_n_veh` | veh | Autos promedio presentes en la celda |
| `{gt,obs}_densidad` | veh/km/carril | Calculada sobre **toda** la longitud de carril de la celda |
| `{gt,obs}_vel` | km/h | Velocidad espacial media (distancia total / tiempo total). NaN si no hubo autos; se recomienda rellenar con `v_ff` |
| `{gt,obs}_flujo` | veh/h/carril | Densidad × velocidad |
| `{gt,obs}_ocupacion` | % | Ocupación |
| `{gt,obs}_perdida`, `{gt,obs}_espera` | s | Segundos acumulados de pérdida de tiempo y de espera |
| `v_ff` | km/h | Velocidad libre de la celda (límites ponderados por km-carril) |
| `gt_congestion` | bool | `gt_vel ≤ 0,55·v_ff` |
| `incidente_activo`, `inc_id` | | El intervalo se solapa con un incidente real en la celda |

**Columnas de `corredor_1min`:**
- Medidas de los detectores E1 del punto medio de cada carril, agregadas por arista: `{gt,obs}_n` (autos que pasaron), `_flujo` (veh/h/carril), `_vel` (media armónica, km/h), `_ocupacion` (%) y `_densidad` (q/v, veh/km/carril).
- Etiquetas:
  - `gt_critico`: criterio del proyecto, `v ≤ 0,55·v_ff` **y** `ρ ≥ 28 veh/km/carril`;
  - `incidente_en_arista`: el incidente está en esa misma arista;
  - `incidente_en_celda`: el incidente está en la misma celda de 200 m.

**Etiqueta de AID:** la etiqueta principal es la del **registro de incidentes** (`incidente_en_*`), no `gt_critico`. La congestión recurrente de hora pico también cumple los umbrales, y separarla de un incidente es precisamente lo que debe aprender el clasificador. Usar los umbrales como etiqueta dispararía la tasa de falsas alarmas (FAR).

## 7. Métricas y evaluación (recomendación del profesor)

**a) Validación del escenario y del dataset** (`etapa2/validar_dataset.py`, en `etapa2/resultados/validacion/`):
- **Por día:** viajes completados, VKT (veh·km), VHT (veh·h), velocidad media de viaje, duración y pérdida de tiempo medias, teletransportes, máximo de autos en la red y de la cola de inserción, y % de celdas congestionadas en el pico de la mañana (`resumen_dias.csv`).
- **Figuras:**
  1. perfiles diarios por tipo de día;
  2. mapas de v/v_libre en valle y en picos;
  3. ejemplo de incidente en el detector y en la celda;
  4. vista V2X frente a la real (correlación);
  5. distribución de viajes.

**b) Predicción a 1 h (Etapa 3)**, por celda, sobre la velocidad (y la densidad) en t+60 min:
- **Error:** MAE, RMSE y MAPE (este solo en celdas con `gt_n_veh` ≥ 1, para evitar dividir por números pequeños).
- **Desgloses:** hora pico frente a valle, tipo de día, celdas de corredor frente al resto, y días con frente a sin incidente.
- **Líneas base obligatorias:**
  - persistencia ($\hat v_{t+60} = v_t$);
  - promedio histórico de la misma hora y el mismo tipo de día (calculado solo con train).
  
  El modelo vale si supera a ambas.
- **Robustez:** evaluar con `obs_*` como entrada y `gt_*` como objetivo. En `test_estres` (penetración del 30 % y 70 %, demanda ×1,12) se mide cuánto se degrada.

**c) AID (Etapa 3):**
- **DR** (tasa de detección): incidentes detectados / incidentes.
- **FAR:** falsas alarmas / decisiones sin incidente. El objetivo del proyecto es ≤ 3–5 %.
- **MTTD:** tiempo medio entre el inicio real (`t_ini` de `incidentes.parquet`) y la primera alarma.
- **Curva DR–FAR** al variar el umbral, desglosada por corredor y severidad (parcial o total).

**d) Bucle SIL (Etapas 4 y 5)**, comparando el escenario base sin IA con el proactivo con V2X, con el **mismo día y la misma semilla**:
- tiempo medio de viaje y pérdida de tiempo (`tripinfo`);
- VHT;
- % de celdas en congestión crítica;
- teletransportes;
- MTTD y FAR en línea.

## 8. Guía para quien vaya a entrenar los modelos

Esta sección es para quien toma el dataset y entrena los modelos de la Etapa 3. No hace falta volver a correr SUMO: todo lo necesario está en `dataset/`.

### 8.1 Reglas que no se pueden romper

1. **Entrada `obs_*`, objetivo `gt_*`.** El modelo solo puede ver lo que reportan los autos conectados (`obs_*`). La vista `gt_*` (todos los autos) se usa solo como objetivo y para evaluar. Entrenar con `gt_*` como entrada da resultados que no se pueden reproducir en el bucle V2X de la Etapa 4.
2. **Dividir por días, usando la columna `split`** de `dias.parquet` o de `calendario.json`: `train` (días 1–28), `val` (29–35) y `test` (36–42). Nunca se mezclan ventanas de un mismo día entre splits.
3. **`test_estres` (días 43–46) no se usa para entrenar ni para ajustar nada.** Sirve solo para medir la robustez al final.
4. **Las estadísticas de normalización, el promedio histórico y los umbrales se calculan solo con `train`.** Si se calculan con todo el dataset, se filtra información del test.
5. **No usar como entrada ninguna columna de etiqueta o de configuración del simulador:**
   - `incidente_activo`, `inc_id`, `incidente_en_*`, `gt_congestion` y `gt_critico`;
   - `factor_demanda`, `factor_efectivo`, `semilla` y `reintentos`.

   Sí se pueden usar la hora, el día de la semana, el tipo de día y la lluvia, porque en la vida real se conocen.
6. **Las ventanas no cruzan días.** Cada día es una simulación independiente que empieza con la red vacía a las 00:00. El día 5 no es la continuación del día 4.
7. **No se pueden comparar las cantidades `obs_*` entre días con distinta penetración V2X.** `obs_n_veh`, `obs_flujo` y `obs_densidad` cuentan solo autos conectados, así que escalan con la penetración: ~50 % en los días normales, 30 % y 70 % en dos días de estrés. Las velocidades (`obs_vel`) y la relación `obs_vel/v_ff` no dependen de la penetración. Si se usan conteos, conviene dividirlos por la penetración estimada, o el modelo fallará en los días de 30 % y 70 %.

### 8.2 Cómo leer los datos

`entrega/cargar_datos.py` (en el zip, en la raíz) ya implementa todo lo de esta sección y está probado: tensores de predicción, ventanas, malla para la CNN, grafo para la GNN, tabla AID y métricas por incidente. El código de abajo es solo lo mínimo para leer las tablas.

```python
import pandas as pd, numpy as np, json

DS = 'dataset'
dias   = pd.read_parquet(f'{DS}/dias.parquet')          # 1 fila por día: split, tipo_dia, lluvia, penetracion_v2x, estadísticas
celdas = pd.read_parquet(f'{DS}/celdas.parquet')        # celda, fila, col, v_ff, lane_km, n_aristas, corredor
incs   = pd.read_parquet(f'{DS}/incidentes.parquet')    # verdad de terreno de cada incidente (t_ini, t_fin, edge, celda_id, severidad)

train_ids = dias.loc[dias.split == 'train', 'dia_id'].tolist()
x = pd.read_parquet(f'{DS}/celdas_5min', filters=[('dia_id', 'in', train_ids)])
```

- **Malla:** celdas de 200 m × 200 m, con 100 columnas y 61 filas. Solo las **1.694 celdas activas** (las que tienen vías) aparecen en las tablas. La posición de cada una está en `celdas.parquet` (`fila`, `col`).
- **Tiempo:** `t` está en segundos desde las 00:00 y marca el **inicio** del intervalo. Un día tiene 288 intervalos de 5 min en `celdas_5min` y 1.440 minutos en `corredor_1min`.
- **Topología de la red:** `escenario/red_info.json` tiene cada arista con `from` y `to` (nodos), `len`, `lanes`, `speed` y `celda`. Sirve para construir el grafo de la GNN y para saber qué arista va antes o después de otra en un corredor.

### 8.3 Preparación para la predicción a 1 h (CNN-LSTM / GNN-LSTM)

**1. Valores faltantes (NaN).** `obs_vel` es NaN cuando ningún auto conectado pasó por la celda en ese intervalo. Pasa en ~30 % de los registros, sobre todo de noche y en calles poco usadas.
- Rellena `obs_vel` con `v_ff` (sin autos se asume flujo libre).
- Agrega un canal de máscara `obs_hay_dato = obs_n_veh > 0` para que el modelo sepa qué valores son medidos y cuáles rellenados.
- **No rellenes con 0:** 0 km/h significa atasco total, lo contrario de "no hay autos".
- En el objetivo `gt_vel`, rellena igual con `v_ff` o excluye esos puntos de la pérdida (`loss` con máscara).

**2. Canales recomendados por celda y por intervalo:**
- `obs_vel / v_ff` (velocidad relativa: queda entre 0 y ~1,2 y es comparable entre celdas);
- `obs_ocupacion`;
- `obs_densidad` y `obs_flujo`, divididos por la penetración (ver regla 7);
- `obs_hay_dato`;
- hora como `sin(2π·t/86400)` y `cos(2π·t/86400)`, día de la semana, tipo de día (one-hot) y lluvia (0/1).

Normaliza cada canal con media y desviación calculadas solo sobre `train`.

**3. Forma del tensor:**
- **CNN-LSTM:** `[días, 288, C, 61, 100]`, ubicando cada celda en `(fila, col)`. Las celdas sin vías quedan en 0 y llevan una máscara fija de "celda activa" para que no cuenten en la pérdida.
- **GNN-LSTM:** `[días, 288, 1.694, C]`, con un grafo de adyacencia. La opción simple es usar las 8 celdas vecinas entre las activas. La opción más fiel es conectar dos celdas si una arista sale de una y llega a la otra (con `from`/`to` de `red_info.json`).

**4. Ventanas deslizantes, dentro de cada día:**
- Entrada: los 12 intervalos anteriores (1 h). Objetivo: `gt_vel` (y opcionalmente `gt_densidad`) en t+60 min, o la secuencia t+5 … t+60.
- Salen 288 − 12 − 12 + 1 = **265 ventanas por día**: ~7.400 para entrenar, ~1.850 para validar y ~1.850 para el test.
- Puedes descartar las ventanas de madrugada (00:00–05:00), porque casi no hay tráfico e inflan las métricas sin aportar.

**5. Líneas base.** Calcula primero la persistencia y el promedio histórico (misma hora y mismo tipo de día, solo con `train`). Si el modelo no las supera, algo está mal.

### 8.4 Preparación para la detección de incidentes (AID)

**1. Unidad de decisión.** Una arista de corredor en un minuto (`corredor_1min`): ~311 aristas × 1.440 minutos por día.

**2. Etiqueta.** Usa `incidente_en_arista` como etiqueta estricta, o `incidente_en_celda`, que también marca las aristas vecinas dentro de la misma celda de 200 m y es más fácil de aprender. **No uses `gt_critico` como etiqueta** (ver §6).

**3. Desbalance extremo.** Solo ~0,015–0,02 % de los registros arista × minuto son positivos con `incidente_en_arista` y ~0,03–0,04 % con `incidente_en_celda`.
- Usa pesos por clase o focal loss.
- Si haces submuestreo de negativos, hazlo **solo en `train`**. Val y test se evalúan con la proporción real, porque si no la FAR sale falsamente baja.

**4. Características útiles,** calculadas por arista con datos `obs_*`:
- valores actuales: `obs_vel/v_ff`, `obs_ocupacion` y `obs_flujo` (este dividido por la penetración);
- su cambio en los últimos 1, 3 y 5 min;
- diferencia con la arista de **aguas arriba** y la de **aguas abajo**. Con un incidente se forma cola antes y la vía queda vacía después; es la señal clásica de los algoritmos tipo California. El orden de las aristas se obtiene de `from`/`to` en `red_info.json`.
- desviación respecto al promedio histórico de esa arista a esa hora y en ese tipo de día (calculado con `train`). Esto es lo que separa un incidente de la congestión normal de hora pico.
- **`obs_n = 0` es información, no un dato perdido.** Si de repente no pasa ningún auto en hora pico, puede ser un cierre total. No rellenes `obs_vel` con `v_ff` en este caso: usa la máscara y deja que el modelo vea el `obs_n = 0`.

**5. Ventana de entrada.** Basta con los últimos 5–15 minutos por arista. Para LSTM o CNN-1D, usa secuencias cortas por arista y comparte los pesos entre aristas.

**6. Evaluación por incidente, no solo por minuto:**
- **DR:** un incidente cuenta como detectado si hay al menos una alarma en su arista o celda entre `t_ini` y `t_fin` de `incidentes.parquet`.
- **MTTD:** primera alarma menos `t_ini`.
- **FAR:** alarmas en minutos sin incidente / minutos sin incidente.
- Exigir 2–3 minutos seguidos de alarma antes de declarar un incidente baja mucho la FAR.

**7. Tamaño de la muestra.** Hay 154 incidentes, pero solo 111 están en los corredores y aparecen en `corredor_1min` (train 58, val 22, test 17, estrés 14). Los otros 43 están en vías fuera de los corredores y solo se ven en `celdas_5min`. Con 17 incidentes en el test, así que las métricas de AID tienen una incertidumbre grande. Repórtalas con intervalo de confianza (por ejemplo, con bootstrap por incidente). Si hace falta más, se pueden generar días adicionales enfocados en incidentes (ver §8.7).

### 8.5 Orden de trabajo sugerido

1. Cargar, revisar `dias.parquet` y las figuras de `etapa2/resultados/validacion/`.
2. Calcular las líneas base (persistencia y promedio histórico) sobre `val`.
3. Entrenar con `train`, elegir hiperparámetros y el umbral de alarma con `val`.
4. Evaluar **una sola vez** sobre `test`, desglosando por tipo de día, por hora pico frente a valle, y por corredor frente al resto.
5. Evaluar sobre `test_estres` para medir robustez:
   - día 43: demanda nominal ×1,12, efectiva ×1,064;
   - día 44: nominal ×1,07 con lluvia, efectiva ×1,057 (incluye el ×1,04 de la lluvia);
   - días 45 y 46: penetración del 30 % y del 70 %.

### 8.6 Errores comunes

| Error | Consecuencia |
|---|---|
| Dividir al azar por ventanas | Fuga de información: el test sale demasiado bueno |
| Normalizar con todo el dataset | Fuga leve del test hacia el entrenamiento |
| Rellenar `obs_vel` con 0 | El modelo confunde "sin autos" con "atasco" |
| Usar `gt_*` como entrada | No funciona en el bucle V2X real |
| Usar conteos `obs_*` sin corregir la penetración | Falla en los días del 30 % y del 70 % |
| Evaluar AID con negativos submuestreados | La FAR reportada es falsamente baja |
| Ventanas que cruzan de un día a otro | Mezcla simulaciones independientes |

### 8.7 ¿Alcanza el dataset?

- **Predicción a 1 h: sí.** Son 46 días × 288 intervalos × 1.694 celdas, unas 12.000 ventanas de toda la ciudad. Es un tamaño comparable a PEMS04 (59 días) y PEMS08 (62 días), con los que se publican modelos GNN-LSTM. El punto débil es que hay pocos fines de semana (~12) y pocos días de lluvia (7), así que el modelo será menos preciso en ellos; por eso se reporta el error separado por tipo de día.
- **AID: alcanza para un primer modelo,** pero con 17 incidentes de corredor en el test las métricas son poco precisas. Si hace falta más, lo recomendable es generar unos 30 días extra solo con el tramo de 06:00 a 21:00 y 8 a 10 incidentes cada uno. Suman 250–300 incidentes más, tardan unas 5–6 h y ocupan unos 600–800 MB. Para eso habría que agregar a `generar_dataset.py` una opción para ese tipo de día (todavía no existe).

## 9. Resultado final del dataset (generado el 9 de noviembre de 2026)

Comando: `python etapa2/generar_dataset.py --todos --paralelo 9`. Empezó a las 23:51 del 8 de noviembre y terminó a las 07:57 del 9 (~8 h). Corrieron los 46 días, sin errores.

| Indicador | Valor |
|---|---|
| Días | 46: 34 laborables (7 con lluvia), 6 sábados y 6 domingos |
| Viajes completados | 8.925.015 (solo autos) |
| Incidentes | 154 planificados y 154 ocurridos: 115 parciales y 39 totales |
| Incidentes por split | train 79, val 29, test 26, test_estres 20 |
| Incidentes por corredor | C1 33, C2 36, C3 42, fuera de corredor 43 |
| Teletransportes | 13.665 (0,15 % de los viajes) |
| Máximo de autos en la red | 4.411 (día 26, laborable con lluvia y 4 incidentes) |
| Tamaño | 1,4 GB: `celdas_5min` 866 MB, `corredor_1min` 303 MB, `viajes` 173 MB |

**Velocidad media de viaje por tipo de día:**

| Tipo | Mínima | Media | Máxima |
|---|---|---|---|
| Laborable | 27,9 | 34,7 | 38,1 km/h |
| Sábado | 38,4 | 39,0 | 39,2 km/h |
| Domingo | 39,3 | 39,5 | 39,6 km/h |

**Reintentos por el vigilante de colapso (§3).** Tres días pasaron de 10.000 autos en la red y se repitieron una vez con 5 % menos de demanda. El factor que realmente se usó queda en `dias.parquet` (`factor_efectivo`).

| Día | Factor nominal | Factor efectivo | Resultado |
|---|---|---|---|
| 38 (test, lluvia) | 1,036 | 1,024 | 29,3 km/h, 918 teletransportes |
| 43 (estrés) | 1,12 | 1,064 | 32,6 km/h, 883 teletransportes |
| 44 (estrés, lluvia) | 1,07 | 1,057 | 28,4 km/h, 1.334 teletransportes |

Los otros 43 días no necesitaron reintentos.

### 9.1 Validación (`etapa2/resultados/validacion/`)

Se corre con `python etapa2/validar_dataset.py` y tarda menos de 10 s.

- **`01_perfiles_diarios.png`:** en los laborables hay tres picos (≈ 08:00, 13:30 y 18:30), con hasta ~3.200 autos en promedio y la velocidad bajando a ~29–31 km/h. El sábado tiene una sola loma al mediodía y el domingo es el más bajo. La cola de inserción no pasa de 6 autos, así que la demanda siempre logra entrar.
- **`02_mapas_velocidad.png`:** a las 03:00 todo está cerca del flujo libre. En los picos la congestión se concentra en el centro y en un núcleo al nororiente del centro, y no se extiende a toda la ciudad, que es lo esperable sin rerouting.
- **`03_ejemplo_incidente.png`:** un cierre total en la Av. Fray Vicente Solano baja la velocidad de la celda de ~35 a ~4 km/h durante el incidente y se recupera en ≤ 5 min al terminar. La vista V2X sigue a la real.
- **`04_v2x_vs_real.png`:** la velocidad V2X por celda tiene una correlación r = 0,988 con la real. Los conteos V2X son ≈ 50 % de los reales, como corresponde a la penetración.
- **`05_viajes.png`:** viaje medio de 6,1 km y 11,5 min. La pérdida de tiempo media sube de ~1,6 min de noche a ~6 min en el pico de la mañana.
- **`resumen_dias.csv`:** las estadísticas de cada día (VKT, VHT, pérdida, % de celdas congestionadas, etc.).

**Conclusión:** el dataset es coherente con una ciudad intermedia andina.
- La demanda sigue un patrón de tres picos.
- Hay diferencia clara entre laborables y fines de semana.
- La lluvia reduce la velocidad entre 3 y 5 km/h.
- Los incidentes tienen un efecto local y se recuperan.
- No hay gridlock: los teletransportes son el 0,15 % de los viajes.
- El máximo de ~4.400 autos simultáneos es coherente con la ley de Little: 17.000 veh/h × ~0,2–0,25 h por viaje.

## 10. Cómo reproducir todo

Se corre desde `Proyecto Interciclo/`, con la imagen Docker `van3twin:jammy` (SUMO 1.12) y el entorno `~/venv_cpu` (pandas, pyarrow, matplotlib).

```bash
# 1. Red (netconvert + poda + unión de nodos)
bash red/construir_red.sh

# 2. Información de la red para Python (aristas, celdas, corredores)
docker run --rm -u $(id -u):$(id -g) -v "$PWD":/w -w /w van3twin:jammy \
    python3 escenario/exportar_red.py red/cuenca_urbana.net.xml escenario/red_info.json

# 3. Mapa PDF de 4 láminas
~/venv_cpu/bin/python escenario/mapa_pdf.py

# 4. Barrido de saturación V0; escribe etapa2/resultados/V0.txt
~/venv_cpu/bin/python etapa2/barrido_saturacion.py \
    --tasas 4000 8000 12000 14000 15000 16000 17000 18000 20000 24000 \
    --semillas 101 202 303 --paralelo 6

# 5. Dataset completo (~8 h con 9 en paralelo; ~1,4 GB)
~/venv_cpu/bin/python etapa2/generar_dataset.py --todos --paralelo 9
#    solo algunos días:  --dias 3 26

# 6. Validación y figuras
~/venv_cpu/bin/python etapa2/validar_dataset.py

# Diagnóstico de bloqueos: reproduce un día hasta una hora dada
~/venv_cpu/bin/python etapa2/prueba_bloqueo.py 26 19 base
```

- Las semillas son fijas (calendario 2026, cada día con `semilla = 1000 + día`), así que repetir los pasos produce el mismo dataset.
- Cada día usa una carpeta temporal en `trabajo/` (del orden de 100 MB), que se borra al terminar.
