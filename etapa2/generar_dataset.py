#!/usr/bin/env python3
"""Generación masiva del dataset de la Etapa 2 (Cuenca urbana, solo autos).

Cada día = 25 h simuladas (23:00 del día anterior a 24:00; la primera hora es calentamiento).
Pipeline por día: viajes (Poisson no homogéneo con el perfil horario del tipo de día y pico = V0)
-> duarouter -> SUMO sin rerouting con incidentes programados -> Parquet -> se borran los XML.
Los días corren en paralelo (un contenedor Docker por día).

Uso:
  ~/venv_cpu/bin/python etapa2/generar_dataset.py --dias 1           # piloto (día 1)
  ~/venv_cpu/bin/python etapa2/generar_dataset.py --todos --paralelo 8
Reanudable: los días con Parquet ya escrito se saltan.
"""
import argparse, json, shutil, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import comun as C

DS = C.RAIZ / 'dataset'
T_FIN = 86400 + C.CALENTAMIENTO
LLUVIA_DEMANDA = 1.04     # la lluvia sube la demanda de autos un 4 %
LIMITE_EN_RED = 10000     # autos simultáneos que indican gridlock (un pico normal tiene 3.000-5.000)
MAX_INTENTOS = 3


# ----------------------------------------------------------------------------- calendario
def calendario(v0, semilla=2026):
    """42 días = 6 semanas sintéticas (lun-vie laborable, sábado, domingo) + 4 días de estrés.
    Semanas 1-4 -> train, 5 -> val, 6 -> test, estrés -> test_estres."""
    rng = np.random.default_rng(semilla)
    dias = []
    for d in range(42):
        semana, dow = d // 7, d % 7
        tipo = 'laborable' if dow < 5 else ('sabado' if dow == 5 else 'domingo')
        lluvia = bool(rng.random() < 0.20)
        con_inc = bool(rng.random() < 0.70)
        dias.append({'dia_id': d + 1, 'semana': semana + 1, 'dow': dow, 'tipo_dia': tipo,
                     'factor_demanda': round(float(rng.uniform(0.88, 1.07)), 3), 'lluvia': lluvia,
                     'penetracion_v2x': 0.5, 'n_incidentes': int(rng.integers(3, 7)) if con_inc else 0,
                     'split': 'train' if semana < 4 else ('val' if semana == 4 else 'test')})
    estres = [('laborable', 1.12, False, 0.5), ('laborable', 1.07, True, 0.5),
              ('laborable', 1.00, False, 0.3), ('laborable', 1.00, False, 0.7)]
    for k, (tipo, f, ll, pen) in enumerate(estres):
        dias.append({'dia_id': 43 + k, 'semana': 7, 'dow': k, 'tipo_dia': tipo, 'factor_demanda': f, 'lluvia': ll,
                     'penetracion_v2x': pen, 'n_incidentes': 5, 'split': 'test_estres'})
    for d in dias:
        d['semilla'] = 1000 + d['dia_id']
        d['v0'] = v0
    return dias


# ----------------------------------------------------------------------------- un día
def correr_dia(dia):
    i = dia['dia_id']
    trabajo = C.RAIZ / 'trabajo' / f'dia_{i:02d}'
    shutil.rmtree(trabajo, ignore_errors=True); trabajo.mkdir(parents=True)
    log = trabajo / 'log.txt'
    t0 = time.time()
    # Si la red colapsa (gridlock que no se disuelve: >LIMITE_EN_RED autos acumulados), la corrida se corta
    # y se repite con 5 % menos de demanda; el factor efectivo queda registrado en el resumen del día.
    for intento in range(MAX_INTENTOS):
        rng = np.random.default_rng(dia['semilla'])
        factor = dia['factor_demanda'] * (LLUVIA_DEMANDA if dia['lluvia'] else 1.0) * 0.95 ** intento
        tasas = C._tasas_por_segundo(C.PERFILES[dia['tipo_dia']], dia['v0'], factor, T_FIN)
        C.PENETRACION_V2X = dia['penetracion_v2x']
        n_viajes = C.generar_viajes(trabajo / 'viajes.trips.xml', tasas, dia['semilla'])
        C.escribir_vtypes(trabajo / 'vtypes.add.xml', 0.88 if dia['lluvia'] else 1.0)
        C.escribir_salidas(trabajo / 'salidas.add.xml', con_corredores=True)
        C.duarouter(trabajo, dia['semilla'], log)
        (trabajo / 'viajes.trips.xml').unlink()
        incs = C.plan_incidentes(i, dia['n_incidentes'], rng, C.uso_rutas(trabajo / 'rutas.rou.xml.gz'))
        C.escribir_incidentes(trabajo / 'incidentes.rou.xml', incs)
        t1 = time.time()
        try:
            C.correr_sumo(trabajo, dia['semilla'], T_FIN, extra_rutas=['/t/incidentes.rou.xml'], log_file=log,
                          limite_en_red=LIMITE_EN_RED)
            break
        except C.Colapso as e:
            C.log(f"día {i:2d}: {e} (factor {factor:.3f}); se repite con 5 % menos de demanda")
            if intento == MAX_INTENTOS - 1:
                raise
            for f in trabajo.iterdir():
                if f.name != 'log.txt':
                    f.unlink()
    t2 = time.time()
    (trabajo / 'rutas.rou.xml.gz').unlink(missing_ok=True)

    # ---- incidentes: tiempos reales de inicio/fin según stop-output
    st = C.leer_stops(trabajo / 'stops.xml')
    for inc in incs:
        s = st[st.veh.str.startswith(inc['inc_id'] + '_L')]
        inc['t_ini'] = float(s.inicio.min() - C.CALENTAMIENTO) if len(s) else np.nan
        inc['t_fin'] = float(s.fin.max() - C.CALENTAMIENTO) if len(s) else np.nan
        inc['carriles_bloqueados'] = len(inc['carriles']); inc['carriles'] = ','.join(map(str, inc['carriles']))

    def etiquetar(df, col_t, periodo, clave):
        df['incidente_activo'] = False; df['inc_id'] = ''
        for inc in incs:
            if np.isnan(inc['t_ini']):
                continue
            m = (df[clave] == inc['celda_id']) & (df[col_t] < inc['t_fin']) & (df[col_t] + periodo > inc['t_ini'])
            df.loc[m, 'incidente_activo'] = True; df.loc[m, 'inc_id'] = inc['inc_id']
        return df

    # ---- celdas cada 5 min (dos vistas)
    celdas = C.tabla_celdas()
    base = pd.MultiIndex.from_product([np.arange(0, 86400, 300), celdas.celda], names=['t', 'celda']).to_frame(index=False)
    for vista, f in (('gt', 'gt_5min.xml.gz'), ('obs', 'v2x_5min.xml.gz')):
        ed = C.leer_edgedata(trabajo / f); (trabajo / f).unlink()
        g = C.agregar_celdas(ed); g['t'] = g.t_sim - C.CALENTAMIENTO
        g = g[(g.t >= 0) & (g.t < 86400)].drop(columns='t_sim')
        base = base.merge(g.rename(columns={c: f'{vista}_{c}' for c in g.columns if c not in ('t', 'celda')}), on=['t', 'celda'], how='left')
    for c in base.columns:
        if c.endswith(('n_veh', 'densidad', 'flujo', 'ocupacion', 'perdida', 'espera')):
            base[c] = base[c].fillna(0)
    base = base.merge(celdas[['celda', 'v_ff']], on='celda')
    base['gt_congestion'] = (base.gt_vel <= 0.55 * base.v_ff).fillna(False)
    base = etiquetar(base, 't', 300, 'celda')
    base.insert(0, 'dia_id', i)
    base = base.rename(columns={'celda': 'celda_id'})
    _guardar(base, 'celdas_5min', i)

    # ---- corredores cada 1 min: una fila por arista (sentido) y minuto, agregando sus carriles
    ar = C.tabla_aristas().set_index('id')
    cor = None
    for vista, f in (('gt', 'e1_gt.xml.gz'), ('obs', 'e1_v2x.xml.gz')):
        e1 = C.leer_e1(trabajo / f); (trabajo / f).unlink()
        e1['t'] = e1.t_sim - C.CALENTAMIENTO
        e1 = e1[(e1.t >= 0) & (e1.t < 86400)]
        e1['inv'] = np.where(e1.n > 0, e1.n / e1.vel_arm.clip(lower=0.1), 0)
        g = e1.groupby(['t', 'edge']).agg(n=('n', 'sum'), inv=('inv', 'sum'), flujo=('flujo', 'mean'),
                                         ocupacion=('ocupacion', 'mean')).reset_index()
        g['vel'] = np.where(g.n > 0, g.n / g.inv.replace(0, np.nan) * 3.6, np.nan)   # km/h, media armónica
        g['densidad'] = np.where(g.vel > 0, g.flujo / g.vel, 0.0)                      # veh/km/carril (q/v)
        g = g.drop(columns=['inv']).rename(columns={c: f'{vista}_{c}' for c in ('n', 'flujo', 'ocupacion', 'vel', 'densidad')})
        cor = g if cor is None else cor.merge(g, on=['t', 'edge'], how='outer')
    cor = cor.join(ar[['corredor', 'celda', 'lanes', 'speed']], on='edge')
    cor['v_ff'] = cor.pop('speed') * 3.6
    cor = cor.rename(columns={'lanes': 'carriles'})
    cor['gt_critico'] = ((cor.gt_vel <= 0.55 * cor.v_ff) & (cor.gt_densidad >= 28)).fillna(False)
    cor = etiquetar(cor, 't', 60, 'celda').rename(columns={'incidente_activo': 'incidente_en_celda'})
    cor['incidente_en_arista'] = False
    for inc in incs:
        if not np.isnan(inc['t_ini']):
            m = (cor.edge == inc['edge']) & (cor.t < inc['t_fin']) & (cor.t + 60 > inc['t_ini'])
            cor.loc[m, 'incidente_en_arista'] = True
    cor.insert(0, 'dia_id', i)
    cols = ['dia_id', 't', 'corredor', 'edge', 'celda', 'carriles', 'v_ff'] + [c for c in cor.columns if c.startswith(('gt_', 'obs_'))] + \
           ['incidente_en_arista', 'incidente_en_celda', 'inc_id']
    _guardar(cor[cols].rename(columns={'celda': 'celda_id'}).sort_values(['t', 'edge']), 'corredor_1min', i)

    # ---- red completa cada 5 min (summary)
    s = C.leer_summary(trabajo / 'summary.xml.gz'); (trabajo / 'summary.xml.gz').unlink()
    s['t'] = s.t_sim - C.CALENTAMIENTO
    s['llegados'] = s.llegados.diff().fillna(0); s['teleports'] = s.teleports.diff().fillna(0)   # acumulados -> por paso
    s = s[(s.t >= 0) & (s.t < 86400)].copy()
    s['bloque'] = (s.t // 300 * 300).astype(int)
    r5 = s.groupby('bloque').agg(en_red=('en_red', 'mean'), en_espera=('en_espera', 'mean'), detenidos=('detenidos', 'mean'),
                                 vel_media=('vel_media', 'mean'), llegados=('llegados', 'sum'), teleports=('teleports', 'sum')).reset_index()
    r5['vel_media'] *= 3.6
    r5 = r5.rename(columns={'bloque': 't'}); r5.insert(0, 'dia_id', i)
    _guardar(r5, 'red_5min', i)

    # ---- viajes
    ti = C.leer_tripinfo(trabajo / 'tripinfo.xml.gz'); (trabajo / 'tripinfo.xml.gz').unlink()
    for c in ('depart', 'arrival'):
        ti[c] -= C.CALENTAMIENTO
    ti = ti[ti.depart >= 0]; ti.insert(0, 'dia_id', i)
    _guardar(ti, 'viajes', i)

    # ---- resumen del día
    res = dict(dia, factor_efectivo=round(factor, 4), reintentos=intento, n_viajes_generados=n_viajes, n_viajes_completados=len(ti),
               vkt_km=float(ti.longitud.sum() / 1000), vht_h=float(ti.duracion.sum() / 3600),
               duracion_media_s=float(ti.duracion.mean()), perdida_media_s=float(ti.perdida.mean()),
               vel_media_viaje_kmh=float(ti.longitud.sum() / ti.duracion.sum() * 3.6),
               teleports=int(r5.teleports.sum()), max_en_red=float(r5.en_red.max()), max_cola=float(r5.en_espera.max()),
               pct_celdas_congestion_pico=float(base[(base.t >= 7 * 3600) & (base.t < 9 * 3600) & (base.gt_n_veh > 0)].gt_congestion.mean() * 100),
               seg_preparacion=round(t1 - t0), seg_sumo=round(t2 - t1), seg_total=round(time.time() - t0))
    json.dump({'dia': res, 'incidentes': incs}, open(DS / 'dias_json' / f'dia_{i:02d}.json', 'w'), default=float, indent=1)
    shutil.rmtree(trabajo)
    return res


def _guardar(df, tabla, i):
    d = DS / tabla; d.mkdir(parents=True, exist_ok=True)
    if 't' in df:
        df['t'] = df['t'].astype('int32')
    for c in df.columns:
        if df[c].dtype == 'float64':
            df[c] = df[c].astype('float32')
    df.to_parquet(d / f'dia_{i:02d}.parquet', index=False, compression='zstd')


def consolidar():
    """Junta los resúmenes por día en dias.parquet e incidentes.parquet."""
    dias, incs = [], []
    for f in sorted((DS / 'dias_json').glob('dia_*.json')):
        j = json.load(open(f)); dias.append(j['dia']); incs += j['incidentes']
    pd.DataFrame(dias).to_parquet(DS / 'dias.parquet', index=False)
    pd.DataFrame(incs).to_parquet(DS / 'incidentes.parquet', index=False)
    C.tabla_celdas().to_parquet(DS / 'celdas.parquet', index=False)
    C.tabla_aristas().to_parquet(DS / 'aristas.parquet', index=False)
    return pd.DataFrame(dias), pd.DataFrame(incs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dias', type=int, nargs='*', default=[])
    ap.add_argument('--todos', action='store_true')
    ap.add_argument('--paralelo', type=int, default=8)
    ap.add_argument('--v0', type=float, default=None, help='demanda pico (veh/h); por defecto etapa2/resultados/V0.txt')
    a = ap.parse_args()
    v0 = a.v0 or float((C.RAIZ / 'etapa2/resultados/V0.txt').read_text())
    (DS / 'dias_json').mkdir(parents=True, exist_ok=True)
    cal = calendario(v0)
    json.dump(cal, open(DS / 'calendario.json', 'w'), indent=1)
    sel = [d for d in cal if a.todos or d['dia_id'] in a.dias]
    pend = [d for d in sel if not (DS / 'dias_json' / f"dia_{d['dia_id']:02d}.json").exists()]
    C.log(f"V0 = {v0:.0f} veh/h | días pedidos {len(sel)}, pendientes {len(pend)}, en paralelo {a.paralelo}")
    with ProcessPoolExecutor(a.paralelo) as ex:
        fut = {ex.submit(correr_dia, d): d for d in pend}
        for f in as_completed(fut):
            d = fut[f]
            try:
                r = f.result()
                C.log(f"día {r['dia_id']:2d} ({r['tipo_dia']:9s} x{r['factor_demanda']:.2f}{' lluvia' if r['lluvia'] else ''}, "
                      f"{r['n_incidentes']} inc): {r['n_viajes_completados']:,} viajes, v={r['vel_media_viaje_kmh']:.1f} km/h, "
                      f"tp={r['teleports']}, máx en red {r['max_en_red']:.0f}, reintentos {r['reintentos']}, sumo {r['seg_sumo']} s, total {r['seg_total']} s")
            except Exception:
                C.log(f"ERROR día {d['dia_id']}:\n{traceback.format_exc()}")
    dias, incs = consolidar()
    C.log(f"Consolidado: {len(dias)} días, {len(incs)} incidentes -> {DS}")


if __name__ == '__main__':
    main()
