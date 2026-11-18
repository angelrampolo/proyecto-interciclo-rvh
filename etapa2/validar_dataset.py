#!/usr/bin/env python3
"""Validación y estadísticas del dataset (figuras + tabla resumen).
Uso: ~/venv_cpu/bin/python etapa2/validar_dataset.py   -> etapa2/resultados/validacion/*.png, resumen_dias.csv
Métricas: ver DOCUMENTACION.md, sección "Métricas y evaluación"."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / 'etapa2'))
import comun as C

DS = RAIZ / 'dataset'
OUT = RAIZ / 'etapa2/resultados/validacion'; OUT.mkdir(parents=True, exist_ok=True)
COL = {'laborable': '#1f6feb', 'sabado': '#d97706', 'domingo': '#059669'}


def leer(tabla, cols=None, filtro=None):
    return pd.read_parquet(DS / tabla, columns=cols, filters=filtro)


def fig_perfiles(dias):
    r = leer('red_5min').merge(dias[['dia_id', 'tipo_dia', 'split']], on='dia_id')
    r = r[r.split != 'test_estres']
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for tipo, g in r.groupby('tipo_dia'):
        m = g.groupby('t').agg(en_red=('en_red', 'mean'), vel=('vel_media', 'mean'), cola=('en_espera', 'mean'))
        h = m.index / 3600
        ax[0].plot(h, m.en_red, color=COL[tipo], label=tipo); ax[1].plot(h, m.vel, color=COL[tipo], label=tipo)
        ax[2].plot(h, m.cola, color=COL[tipo], label=tipo)
    for a, t, y in zip(ax, ['Autos circulando en la red', 'Velocidad media de la red', 'Cola de inserción (autos esperando entrar)'], ['vehículos', 'km/h', 'vehículos']):
        a.set(title=t, xlabel='Hora del día', ylabel=y, xticks=range(0, 25, 3)); a.grid(alpha=.3); a.legend()
    fig.suptitle('Perfiles diarios promedio por tipo de día', fontweight='bold'); fig.tight_layout()
    fig.savefig(OUT / '01_perfiles_diarios.png', dpi=140); plt.close(fig)


def fig_mapa_velocidad(dias):
    lab = dias[(dias.tipo_dia == 'laborable') & (dias.split == 'train')].dia_id.tolist()[:5]
    cel = leer('celdas_5min', ['dia_id', 't', 'celda_id', 'gt_vel', 'v_ff', 'gt_n_veh'], [('dia_id', 'in', lab)])
    R = C.red_info(); celdas = C.tabla_celdas()
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for a, (h0, tit) in zip(ax, [(3, '03:00 (valle)'), (7.5, '07:30 (pico mañana)'), (18.25, '18:15 (pico tarde)')]):
        sel = cel[(cel.t >= h0 * 3600) & (cel.t < h0 * 3600 + 1800) & (cel.gt_n_veh > 0)]
        m = sel.groupby('celda_id').apply(lambda d: np.average(d.gt_vel / d.v_ff, weights=d.gt_n_veh), include_groups=False)
        grid = np.full((R['ny'], R['nx']), np.nan)
        for c, v in m.items():
            grid[c // R['nx'], c % R['nx']] = v
        im = a.imshow(grid, origin='lower', cmap='RdYlGn', vmin=0.2, vmax=1.0, interpolation='nearest')
        a.set_title(tit); a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=.8, label='v / v_libre')
    fig.suptitle('Estado de la red (vista real, promedio de 5 días laborables)', fontweight='bold')
    fig.savefig(OUT / '02_mapas_velocidad.png', dpi=140, bbox_inches='tight'); plt.close(fig)


def fig_incidente(dias, incs):
    i = incs.dropna(subset=['t_ini'])
    i = i[(i.corredor != 'ninguno') & (i.severidad == 'total')].head(1)
    if not len(i):
        i = incs.dropna(subset=['t_ini']).head(1)
    inc = i.iloc[0]
    cor = leer('corredor_1min', None, [('dia_id', '=', int(inc.dia_id)), ('celda_id', '=', int(inc.celda_id))])
    cel = leer('celdas_5min', None, [('dia_id', '=', int(inc.dia_id)), ('celda_id', '=', int(inc.celda_id))])
    w = (cor.t >= inc.t_ini - 3600) & (cor.t <= inc.t_fin + 3600)
    fig, ax = plt.subplots(1, 2, figsize=(16, 4.8))
    c = cor[w].groupby('t')[['gt_vel', 'obs_vel', 'gt_flujo']].mean()
    ax[0].plot(c.index / 3600, c.gt_vel, label='Real (todos los autos)', color='k')
    ax[0].plot(c.index / 3600, c.obs_vel, label='Observada V2X (50 %)', color='#1f6feb', alpha=.8)
    ax[0].axvspan(inc.t_ini / 3600, inc.t_fin / 3600, color='red', alpha=.15, label='Incidente activo')
    ax[0].set(title=f'Detectores E1 cada 1 min — {inc.nombre_via} ({inc.corredor})', xlabel='Hora', ylabel='km/h'); ax[0].legend(); ax[0].grid(alpha=.3)
    w2 = (cel.t >= inc.t_ini - 3600) & (cel.t <= inc.t_fin + 3600)
    ax[1].plot(cel[w2].t / 3600, cel[w2].gt_vel, 'o-', color='k', label='Real')
    ax[1].plot(cel[w2].t / 3600, cel[w2].obs_vel, 's-', color='#1f6feb', label='V2X')
    ax[1].axvspan(inc.t_ini / 3600, inc.t_fin / 3600, color='red', alpha=.15)
    ax[1].set(title=f'Celda {int(inc.celda_id)} cada 5 min', xlabel='Hora', ylabel='km/h'); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle(f'Ejemplo de incidente {inc.inc_id}: {inc.severidad}, {int(inc.carriles_bloqueados)}/{int(inc.n_carriles_via)} carriles, '
                 f'{(inc.t_fin - inc.t_ini) / 60:.0f} min', fontweight='bold')
    fig.tight_layout(); fig.savefig(OUT / '03_ejemplo_incidente.png', dpi=140); plt.close(fig)


def fig_v2x_vs_real(dias):
    d = dias[dias.split == 'train'].dia_id.tolist()[:3]
    cel = leer('celdas_5min', ['gt_vel', 'obs_vel', 'gt_n_veh', 'obs_n_veh'], [('dia_id', 'in', d)]).dropna()
    cel = cel[cel.gt_n_veh >= 1]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].hexbin(cel.gt_vel, cel.obs_vel, gridsize=60, bins='log', cmap='Blues')
    ax[0].plot([0, 80], [0, 80], 'r--'); ax[0].set(xlabel='Velocidad real (km/h)', ylabel='Velocidad V2X (km/h)',
                                                  title=f'Celdas 5 min: r = {np.corrcoef(cel.gt_vel, cel.obs_vel)[0, 1]:.3f}')
    ax[1].hexbin(cel.gt_n_veh, cel.obs_n_veh, gridsize=60, bins='log', cmap='Oranges', xscale='log', yscale='log')
    ax[1].set(xlabel='Autos en celda (real)', ylabel='Autos en celda (V2X)', title='Muestra V2X ≈ 50 % del tráfico')
    fig.suptitle('Vista observada V2X vs. verdad de terreno', fontweight='bold'); fig.tight_layout()
    fig.savefig(OUT / '04_v2x_vs_real.png', dpi=140); plt.close(fig)


def fig_viajes(dias):
    d = dias[(dias.split == 'train') & (dias.tipo_dia == 'laborable')].dia_id.tolist()[:3]
    v = leer('viajes', ['depart', 'duracion', 'longitud', 'perdida'], [('dia_id', 'in', d)])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].hist(v.longitud / 1000, bins=60, color='#1f6feb'); ax[0].set(title=f'Longitud de viaje (media {v.longitud.mean() / 1000:.1f} km)', xlabel='km')
    ax[1].hist(v.duracion / 60, bins=80, range=(0, 60), color='#d97706'); ax[1].set(title=f'Duración (media {v.duracion.mean() / 60:.1f} min)', xlabel='min')
    h = (v.depart // 3600).astype(int); m = v.groupby(h).perdida.mean() / 60
    ax[2].bar(m.index, m.values, color='#dc2626'); ax[2].set(title='Pérdida de tiempo media por hora de salida', xlabel='Hora', ylabel='min')
    for a in ax:
        a.grid(alpha=.3)
    fig.suptitle('Viajes (días laborables)', fontweight='bold'); fig.tight_layout()
    fig.savefig(OUT / '05_viajes.png', dpi=140); plt.close(fig)


def main():
    dias = pd.read_parquet(DS / 'dias.parquet'); incs = pd.read_parquet(DS / 'incidentes.parquet')
    cols = ['dia_id', 'split', 'tipo_dia', 'factor_demanda', 'lluvia', 'n_incidentes', 'n_viajes_completados', 'vkt_km', 'vht_h',
            'vel_media_viaje_kmh', 'duracion_media_s', 'perdida_media_s', 'teleports', 'max_en_red', 'max_cola', 'pct_celdas_congestion_pico', 'seg_total']
    dias[cols].round(2).to_csv(OUT / 'resumen_dias.csv', index=False)
    print(dias[cols].round(1).to_string(index=False))
    fig_perfiles(dias); fig_mapa_velocidad(dias); fig_incidente(dias, incs); fig_v2x_vs_real(dias); fig_viajes(dias)
    i = incs.dropna(subset=['t_ini'])
    print(f"\nIncidentes: {len(incs)} planificados, {len(i)} ocurrieron; por corredor:\n{i.corredor.value_counts().to_string()}")
    print(f"Figuras en {OUT}")


if __name__ == '__main__':
    main()
