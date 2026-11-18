#!/usr/bin/env python3
"""Reproduce un día del calendario hasta una hora dada con variantes de opciones de SUMO,
para diagnosticar bloqueos (gridlock). Uso: prueba_bloqueo.py DIA HORA_FIN VARIANTE [VARIANTE...]"""
import json, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import comun as C
_dk = C.docker
C.docker = lambda cmd, *a, **k: _dk([x for x in cmd if x != '--no-warnings'], *a, **k)

VARIANTES = {'base': []}   # opciones extra de SUMO para comparar (la configuración base ya está en comun.correr_sumo)

def preparar(dia):
    base = C.RAIZ / 'trabajo' / f'prueba_{dia["dia_id"]:02d}'
    shutil.rmtree(base, ignore_errors=True); base.mkdir(parents=True)
    rng = np.random.default_rng(dia['semilla'])
    factor = dia['factor_demanda'] * (1.05 if dia['lluvia'] else 1.0)
    T_FIN = 86400 + C.CALENTAMIENTO
    tasas = C._tasas_por_segundo(C.PERFILES[dia['tipo_dia']], dia['v0'], factor, T_FIN)
    C.PENETRACION_V2X = dia['penetracion_v2x']
    C.generar_viajes(base / 'viajes.trips.xml', tasas, dia['semilla'])
    C.escribir_vtypes(base / 'vtypes.add.xml', 0.88 if dia['lluvia'] else 1.0)
    open(base / 'salidas.add.xml', 'w').write('<additional/>\n')
    C.duarouter(base, dia['semilla'], base / 'log.txt')
    incs = C.plan_incidentes(dia['dia_id'], dia['n_incidentes'], rng, C.uso_rutas(base / 'rutas.rou.xml.gz'))
    C.escribir_incidentes(base / 'incidentes.rou.xml', incs)
    return base

def correr(args):
    base, dia, hfin, var = args
    t = base.parent / f'{base.name}_{var}'
    shutil.rmtree(t, ignore_errors=True); t.mkdir()
    for f in ['rutas.rou.xml.gz', 'incidentes.rou.xml', 'vtypes.add.xml', 'salidas.add.xml']:
        shutil.copy(base / f, t / f)
    t0 = time.time()
    extra = [a for a in VARIANTES[var]]
    cmd_extra = extra
    # sin --no-warnings para ver motivos de teletransporte: se reemplaza en la lista
    C.correr_sumo(t, dia['semilla'], int((hfin + 1) * 3600), con_tripinfo=False,
                  extra_rutas=['/t/incidentes.rou.xml'], log_file=t / 'log.txt', extra=cmd_extra)
    s = C.leer_summary(t / 'summary.xml.gz')
    s['h'] = s.t_sim / 3600 - 1
    return var, s, round(time.time() - t0)

if __name__ == '__main__':
    d, hfin, vars_ = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3:]
    dia = [x for x in json.load(open(C.RAIZ / 'dataset' / 'calendario.json')) if x['dia_id'] == d][0]
    dia['v0'] = float((C.RAIZ / 'etapa2/resultados/V0.txt').read_text())   # V0 vigente
    base = preparar(dia)
    with ProcessPoolExecutor(len(vars_)) as ex:
        for var, s, seg in ex.map(correr, [(base, dia, hfin, v) for v in vars_]):
            r = s[s.h % 1 == 0] if (s.h % 1 == 0).any() else s
            print(f'== día {d} variante {var} ({seg} s)')
            print(s[(s.t_sim % 3600) == 0][['h', 'en_red', 'detenidos', 'vel_media', 'teleports']].round(1).to_string(index=False), flush=True)
