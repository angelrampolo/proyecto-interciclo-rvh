# Formato de datos y transición al dato real

El proyecto se probó con **datos sintéticos** generados en SUMO, porque los datos de la EMOV EP o de cámaras dependen de terceros y no cubren el periodo de estudio. El código no depende del origen de los datos, sino de este formato.

**Cuando llegue el dato real:**

1. Se convierte a las tablas de abajo.
2. Se revisa con `python v2/validar_datos.py /ruta/al/dato_real`.
3. Se apunta el proyecto a esa carpeta con `export RVH_DATOS=/ruta/al/dato_real`.

Ningún script cambia.

Todas las tablas son Parquet. El tiempo `t` está en segundos desde las 00:00 y marca el **inicio** del intervalo.

## Tablas que usan los modelos

| Tabla | Granularidad | Columnas mínimas | De dónde saldría el dato real |
|---|---|---|---|
| `celdas_5min/dia_NN.parquet` | día × intervalo de 5 min × celda de 200 m | `dia_id, t, celda_id, gt_n_veh, gt_vel, gt_densidad, gt_flujo, gt_ocupacion, obs_n_veh, obs_vel, v_ff` | Velocidades por tramo (sondas GPS, Waze for Cities, buses) agregadas a la malla de 200 m |
| `corredor_1min/dia_NN.parquet` | día × minuto × arista de corredor | `dia_id, t, corredor, edge, celda_id, carriles, v_ff, gt_n, gt_flujo, gt_ocupacion, gt_vel, obs_n, obs_vel, obs_ocupacion` | Detectores de lazo o cámaras de conteo de la EMOV en Américas, Circunvalación Sur y Remigio Crespo |
| `dias.parquet` | un registro por día | `dia_id, tipo_dia, lluvia, penetracion_v2x, split` | Calendario y registro de lluvia (INAMHI) |
| `incidentes.parquet` | un registro por incidente | `inc_id, dia_id, edge, corredor, celda_id, t_ini, t_fin, severidad` | Registro de siniestros (ECU 911, Agentes Civiles de Tránsito) |
| `celdas.parquet`, `aristas.parquet` | estático | `celda, v_ff, fila, col, corredor` / `id, lanes, speed, celda, corredor` | Se generan desde la red de OSM con `escenario/exportar_red.py` |

### Columnas

- **`gt_*`:** lo que mide la infraestructura, con todos los vehículos.
- **`obs_*`:** lo que reportan solo los vehículos conectados (V2X). Con dato real de un solo tipo, se copian las columnas `gt_*` en `obs_*`.
- **`gt_vel`:** velocidad media en km/h. Puede venir vacía (NaN) si no hubo vehículos; el código la rellena con `v_ff`.
- **Unidades:** `gt_ocupacion` en %, `gt_flujo` en veh/h/carril, `gt_densidad` en veh/km/carril.

## Comprobación

```bash
python v2/validar_datos.py                  # dataset sintético
python v2/validar_datos.py /ruta/dato_real  # dato real convertido
```

El validador revisa:

- que estén las tablas;
- las columnas y sus tipos;
- rangos físicos: velocidad de 0 a 150 km/h y ocupación de 0 a 100 %;
- que `t` sea múltiplo de 300 s o de 60 s, según la tabla.

Los resultados obtenidos con el dataset sintético están etiquetados como **datos sintéticos (SUMO)** en el informe, el tablero y las figuras. No son medidas de campo.
