#!/usr/bin/env python3
"""Barrido de demanda para hallar el punto de saturación V0 de la red de Cuenca urbana.

Para cada tasa R (veh/h) se insertan viajes de solo autos a tasa constante durante 75 min
(15 min de llenado + 60 min de medición), con rutas de duarouter y sin rerouting.
En la ventana de medición se calcula:
  - throughput: autos que terminan su viaje por hora
  - eficiencia = throughput / R (en régimen libre ~1)
  - velocidad media de la red, cola de inserción (autos que no logran entrar) y teletransportes.
V0 = mayor demanda (promedio de semillas) con eficiencia >= 0.95, cola < 2 % de R,
teletransportes < 1 % de R y velocidad >= 75 % de la de baja demanda.

Uso: ~/venv_cpu/bin/python etapa2/barrido_saturacion.py --tasas 4000 8000 12000 --paralelo 8
"""
import argparse, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import comun as C

T_LLENADO, T_MEDIDA = 900, 3600


def una_tasa(R, semilla):
    trabajo = C.RAIZ / 'trabajo' / f'barrido_{R}_{semilla}'
    shutil.rmtree(trabajo, ignore_errors=True); trabajo.mkdir(parents=True)
    log = trabajo / 'log.txt'
    t0 = time.time()
    t_fin = T_LLENADO + T_MEDIDA
    n = C.generar_viajes(trabajo / 'viajes.trips.xml', np.full(t_fin, R / 3600.0), semilla)
    C.escribir_vtypes(trabajo / 'vtypes.add.xml')
    open(trabajo / 'salidas.add.xml', 'w').write('<additional/>\n')
    C.duarouter(trabajo, semilla, log)
    C.correr_sumo(trabajo, semilla, t_fin, con_tripinfo=True, log_file=log)
    s = C.leer_summary(trabajo / 'summary.xml.gz')
    ti = C.leer_tripinfo(trabajo / 'tripinfo.xml.gz')
    v = s[(s.t_sim >= T_LLENADO) & (s.t_sim < t_fin)]
    llegados = s.llegados.iloc[-1] - s[s.t_sim < T_LLENADO].llegados.iloc[-1]
    w = ti[(ti.depart >= T_LLENADO)]
    res = {'demanda_vph': R, 'semilla': semilla, 'viajes': n, 'throughput_vph': llegados * 3600 / T_MEDIDA,
           'vel_media_kmh': v.vel_media.mean() * 3.6, 'en_red_medio': v.en_red.mean(),
           'cola_final': int(s.en_espera.iloc[-1]), 'teleports': int(s.teleports.iloc[-1]),
           'perdida_media_s': w.perdida.mean(), 'duracion_media_s': w.duracion.mean(),
           'longitud_media_km': w.longitud.mean() / 1000, 'seg_cpu': round(time.time() - t0)}
    res['eficiencia'] = res['throughput_vph'] / R
    shutil.rmtree(trabajo)
    C.log(f"R={R:>6} veh/h  thr={res['throughput_vph']:7.0f}  efic={res['eficiencia']:.2f}  v={res['vel_media_kmh']:5.1f} km/h  "
          f"cola={res['cola_final']:5d}  tp={res['teleports']:4d}  ({res['seg_cpu']} s)")
    return res


def figura(df, v0, salida, todo=None):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    if todo is not None:
        ax[0].scatter(todo.demanda_vph, todo.throughput_vph, s=12, color='#1f6feb', alpha=.35, label='Cada semilla')
        ax[1].scatter(todo.demanda_vph, todo.vel_media_kmh, s=12, color='#d97706', alpha=.35)
    ax[0].plot(df.demanda_vph, df.throughput_vph, 'o-', color='#1f6feb', label='Throughput medio')
    ax[0].plot(df.demanda_vph, df.demanda_vph, '--', color='gray', label='Demanda (ideal)')
    ax[0].set(xlabel='Demanda (veh/h)', ylabel='Autos que terminan su viaje (veh/h)', title='Throughput vs demanda')
    ax[1].plot(df.demanda_vph, df.vel_media_kmh, 'o-', color='#d97706')
    ax[1].set(xlabel='Demanda (veh/h)', ylabel='km/h', title='Velocidad media de la red')
    ax[2].plot(df.demanda_vph, df.cola_final, 'o-', color='#dc2626', label='Cola de inserción')
    ax[2].plot(df.demanda_vph, df.teleports, 's-', color='#7c3aed', label='Teletransportes')
    ax[2].set(xlabel='Demanda (veh/h)', ylabel='vehículos', title='Cola y teletransportes')
    for a in ax:
        a.axvline(v0, color='k', ls=':', lw=1.2); a.grid(alpha=.3)
    ax[0].legend(); ax[2].legend()
    fig.suptitle(f'Punto de saturación de Cuenca urbana (solo autos, duarouter, sin rerouting): V0 ≈ {v0:,.0f} veh/h', fontweight='bold')
    fig.tight_layout(); fig.savefig(salida, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasas', type=int, nargs='+', default=[4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000])
    ap.add_argument('--paralelo', type=int, default=8)
    ap.add_argument('--semillas', type=int, nargs='+', default=[101])
    a = ap.parse_args()
    C.log(f"Barrido de saturación: {a.tasas} veh/h, {a.paralelo} en paralelo")
    with ProcessPoolExecutor(a.paralelo) as ex:
        pares = [(R, s) for s in a.semillas for R in a.tasas]
        res = list(ex.map(una_tasa, [p[0] for p in pares], [p[1] for p in pares]))
    out = C.RAIZ / 'etapa2' / 'resultados'; out.mkdir(exist_ok=True)
    nuevo = pd.DataFrame(res)
    f = out / 'barrido_saturacion.csv'
    todo = pd.concat([pd.read_csv(f), nuevo]) if f.exists() else nuevo
    if 'semilla' not in todo:
        todo['semilla'] = 101
    todo['semilla'] = todo.semilla.fillna(101).astype(int)
    todo = todo.drop_duplicates(['demanda_vph', 'semilla'], keep='last').sort_values(['demanda_vph', 'semilla'])
    todo.to_csv(f, index=False)
    # promedio por demanda (varias semillas)
    df = todo.groupby('demanda_vph').mean(numeric_only=True).reset_index()
    df['n_semillas'] = todo.groupby('demanda_vph').size().values
    v_libre = df.vel_media_kmh.iloc[0]
    # V0: mayor demanda que la red atiende sin degradarse (criterios sobre el promedio de semillas)
    ok = df[(df.eficiencia >= 0.95) & (df.cola_final < 0.02 * df.demanda_vph) &
            (df.vel_media_kmh >= 0.75 * v_libre) & (df.teleports < 0.01 * df.demanda_vph)]
    v0 = float(ok.demanda_vph.max()) if len(ok) else float(df.demanda_vph.min())
    df['bajo_saturacion'] = df.demanda_vph <= v0
    df.to_csv(out / 'barrido_saturacion_promedio.csv', index=False)
    figura(df, v0, out / 'barrido_saturacion.png', todo)
    (out / 'V0.txt').write_text(f'{v0:.0f}\n')
    C.log(f"V0 = {v0:.0f} veh/h  -> {f}")
    print(df.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
