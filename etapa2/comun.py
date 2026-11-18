"""Núcleo de la Etapa 2: demanda (solo autos), incidentes, ejecución SUMO en Docker
y agregación de salidas a Parquet. Lo usan barrido_saturacion.py y generar_dataset.py.

Convenciones
- Tiempo de simulación t_sim = t_dia + 3600: la simulación arranca a las 23:00 del día
  anterior (1 h de calentamiento que se descarta) y termina a las 24:00.
- Solo autos: tipos auto_humano y auto_v2x (50 % conectados). Los vehículos 'incidente'
  son autos detenidos que bloquean carriles y no se cuentan en las métricas.
- Sin rerouting (recomendación del profesor): las rutas se fijan con duarouter.
"""
import gzip, json, math, os, random, re, shutil, subprocess, time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RED = 'red/cuenca_urbana.net.xml'
IMG = 'van3twin:jammy'
CELDA = 200.0                       # m
CALENTAMIENTO = 3600                # s
PENETRACION_V2X = 0.5
CORREDORES = {
    'C1_Americas': ['Avenida de las Américas'],
    'C2_Circunvalacion_Sur': ['Circunvalación Sur'],
    'C3_RemigioCrespo_Solano': ['Avenida Remigio Crespo Toral', 'Avenida Fray Vicente Solano'],
}
# Fracción de la demanda pico (V0) por hora del día (0..23). Laborable con los tres picos
# típicos de Cuenca (07-08, 12-14 por el almuerzo/escuelas y 18-19).
PERFILES = {
    'laborable': [.06, .04, .03, .03, .06, .18, .55, 1.0, .90, .62, .58, .65, .85, .92, .75, .62, .65, .82, .98, .80, .55, .38, .24, .12],
    'sabado':    [.10, .07, .05, .04, .05, .10, .22, .40, .55, .65, .72, .78, .80, .76, .70, .66, .66, .68, .70, .62, .50, .40, .30, .18],
    'domingo':   [.12, .08, .06, .04, .04, .06, .12, .22, .35, .48, .56, .60, .62, .60, .55, .50, .50, .52, .52, .45, .36, .26, .18, .12],
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- red y malla
_RED = None
def red_info():
    global _RED
    if _RED is None:
        _RED = json.load(open(RAIZ / 'escenario/red_info.json'))
        b = _RED['boundary']
        _RED['nx'] = int(math.ceil((b[2] - b[0]) / CELDA))
        _RED['ny'] = int(math.ceil((b[3] - b[1]) / CELDA))
        for e in _RED['edges']:
            col = min(int((e['mid'][0] - b[0]) // CELDA), _RED['nx'] - 1)
            fila = min(int((e['mid'][1] - b[1]) // CELDA), _RED['ny'] - 1)
            e['celda'] = fila * _RED['nx'] + col
            e['corredor'] = next((c for c, nombres in CORREDORES.items() if e['name'] in nombres), None)
        _RED['por_id'] = {e['id']: e for e in _RED['edges']}
    return _RED


def tabla_aristas():
    r = red_info()
    return pd.DataFrame([{k: e[k] for k in ('id', 'name', 'type', 'len', 'lanes', 'speed', 'celda', 'corredor')}
                         for e in r['edges']])


# ----------------------------------------------------------------------------- docker
def docker(cmd, trabajo, log_file=None):
    """Ejecuta un comando en un contenedor efímero van3twin:jammy.
    /w = proyecto (solo lectura de la red), /t = carpeta de trabajo del día."""
    full = ['docker', 'run', '--rm', '--ulimit', 'core=0', '-u', f'{os.getuid()}:{os.getgid()}',
            '-e', 'SUMO_HOME=/usr/share/sumo', '-v', f'{RAIZ}:/w:ro', '-v', f'{trabajo}:/t', '-w', '/w', IMG] + cmd
    with open(log_file or os.devnull, 'a') as f:
        r = subprocess.run(full, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"falló {' '.join(cmd[:2])} (código {r.returncode}); ver {log_file}")


# ----------------------------------------------------------------------------- demanda
def _tasas_por_segundo(perfil, v0, factor, t_fin):
    """Tasa de llegadas (veh/s) segundo a segundo. Constante dentro de cada hora (el promedio de la
    hora pico es exactamente V0 x factor) con transiciones suavizadas por media móvil de 20 min.
    El índice es t_sim; t_sim 0..3599 corresponde a las 23:00 del día anterior (perfil[23])."""
    t = np.arange(t_fin)
    hora = ((t - CALENTAMIENTO) // 3600) % 24
    tasa = np.asarray(perfil)[hora] * v0 * factor / 3600.0
    k = 1200
    pad = np.concatenate([np.full(k // 2, tasa[0]), tasa, np.full(k // 2, tasa[-1])])
    return np.convolve(pad, np.ones(k) / k, mode='same')[k // 2:k // 2 + t_fin]


def generar_viajes(destino, tasas, semilla, prefijo=''):
    """Escribe trips de solo autos con un proceso de Poisson no homogéneo.
    Orígenes/destinos ponderados por longitud x carriles (como randomTrips), bordes de la red x3
    (tráfico que entra/sale de la ciudad) y decaimiento con la distancia (gravedad):
    P(aceptar) = exp(-(d-1000)/5000), con d >= 1 km en línea recta."""
    rng = np.random.default_rng(semilla)
    r = red_info()
    E = r['edges']
    peso = np.array([e['len'] * e['lanes'] * (3.0 if e['fringe'] else 1.0) for e in E])
    peso /= peso.sum()
    mids = np.array([e['mid'] for e in E])
    cuentas = rng.poisson(tasas)
    deps = np.repeat(np.arange(len(tasas)), cuentas) + rng.random(int(cuentas.sum()))
    deps.sort()
    n = len(deps)
    o = np.empty(n, int); d = np.empty(n, int); pend = np.arange(n)
    while len(pend):
        oo = rng.choice(len(E), len(pend), p=peso); dd = rng.choice(len(E), len(pend), p=peso)
        dist = np.hypot(*(mids[oo] - mids[dd]).T)
        ok = (dist >= 1000) & (rng.random(len(pend)) < np.exp(-(dist - 1000) / 5000))
        o[pend[ok]] = oo[ok]; d[pend[ok]] = dd[ok]; pend = pend[~ok]
    v2x = rng.random(n) < PENETRACION_V2X
    with open(destino, 'w') as f:
        f.write('<routes>\n')
        for i in range(n):
            f.write(f'  <trip id="{prefijo}{i}" type="{"auto_v2x" if v2x[i] else "auto_humano"}" depart="{deps[i]:.2f}" '
                    f'from="{E[o[i]]["id"]}" to="{E[d[i]]["id"]}" departLane="best" departSpeed="max" departPos="random"/>\n')
        f.write('</routes>\n')
    return n


def escribir_vtypes(destino, factor_vel=1.0):
    """Mismo comportamiento para humano y V2X (conectado != automatizado): solo cambia la etiqueta,
    así la vista V2X es una muestra aleatoria del 50 % del tráfico. Lluvia: factor_vel < 1."""
    sf = f'normc({factor_vel:.2f},0.10,0.20,2.00)'
    comun = f'vClass="passenger" length="4.5" minGap="2.5" accel="2.6" decel="4.5" emergencyDecel="9" sigma="0.5" tau="1.0" speedFactor="{sf}"'
    open(destino, 'w').write(f'''<additional>
    <vType id="auto_humano" {comun} color="0.6,0.6,0.6"/>
    <vType id="auto_v2x" {comun} color="0,0.6,1"/>
    <vType id="incidente" vClass="passenger" length="4.5" minGap="1.0" color="1,0,0"/>
</additional>
''')


def escribir_salidas(destino, con_corredores=True):
    """edgeData cada 300 s en dos vistas (real = todos los autos, v2x = solo conectados) y
    detectores E1 por carril en los corredores cada 60 s (mismas dos vistas)."""
    attrs = 'sampledSeconds speed occupancy timeLoss waitingTime entered'
    lineas = ['<additional>',
              f'    <edgeData id="gt" freq="300" file="/t/gt_5min.xml.gz" excludeEmpty="true" vTypes="auto_humano auto_v2x" writeAttributes="{attrs}"/>',
              f'    <edgeData id="v2x" freq="300" file="/t/v2x_5min.xml.gz" excludeEmpty="true" vTypes="auto_v2x" writeAttributes="{attrs}"/>']
    if con_corredores:
        for e in red_info()['edges']:
            if e['corredor'] and e['len'] >= 30:
                for l in range(e['lanes']):
                    pos = round(e['len'] / 2, 1)
                    lineas.append(f'    <inductionLoop id="gt|{e["id"]}|{l}" lane="{e["id"]}_{l}" pos="{pos}" freq="60" file="/t/e1_gt.xml.gz" vTypes="auto_humano auto_v2x"/>')
                    lineas.append(f'    <inductionLoop id="v2x|{e["id"]}|{l}" lane="{e["id"]}_{l}" pos="{pos}" freq="60" file="/t/e1_v2x.xml.gz" vTypes="auto_v2x"/>')
    lineas.append('</additional>')
    open(destino, 'w').write('\n'.join(lineas) + '\n')


def escribir_incidentes(destino, incidentes):
    """Cada incidente = 1 auto detenido por carril bloqueado (stop en vía, no estacionamiento)."""
    vs = []
    for inc in incidentes:
        e = red_info()['por_id'][inc['edge']]
        for l in inc['carriles']:
            dep = inc['t_ini_plan'] + CALENTAMIENTO - 20
            vs.append((dep, f'''    <vehicle id="{inc['inc_id']}_L{l}" type="incidente" depart="{dep:.1f}" departLane="{l}" departPos="{max(inc['pos'] - 15, 0):.1f}" departSpeed="0">
        <route edges="{e['id']}"/>
        <stop lane="{e['id']}_{l}" endPos="{inc['pos']:.1f}" duration="{inc['duracion']}"/>
    </vehicle>'''))
    vs.sort()
    open(destino, 'w').write('<routes>\n' + '\n'.join(v for _, v in vs) + '\n</routes>\n')


def uso_rutas(path):
    """Cuántas rutas de duarouter pasan por cada arista (exposición al tráfico)."""
    from collections import Counter
    c = Counter()
    for r in _iter(path, 'route'):
        c.update(r.get('edges').split())
    return c


def plan_incidentes(dia_id, n, rng, uso, t_min=5 * 3600, t_max=22 * 3600):
    """70 % en los corredores (el corredor se elige con igual probabilidad entre C1, C2 y C3) y
    30 % en otras vías principales (>= 50 km/h, >= 120 m). Dentro de cada grupo la arista se elige
    con probabilidad proporcional a su uso (rutas que la cruzan), entre las que superan la mediana:
    los accidentes ocurren donde hay tráfico.
    Severidad (vías de varios carriles): 75 % bloquea 1 carril, 15 % bloquea 2 (si hay >= 3)
    y 10 % cierre total. En vías de 1 carril el bloqueo es siempre total."""
    r = red_info()
    def pool(f):
        es = [e for e in r['edges'] if f(e) and uso.get(e['id'], 0) > 0]
        u = np.array([uso[e['id']] for e in es], float)
        m = u >= np.median(u)
        es = [e for e, k in zip(es, m) if k]; u = u[m]
        return es, u / u.sum()
    grupos = {c: pool(lambda e, c=c: e['corredor'] == c and e['len'] >= 120) for c in CORREDORES}
    otras = pool(lambda e: not e['corredor'] and e['speed'] >= 13.8 and e['len'] >= 120)
    out = []
    for k in range(n):
        es, p = grupos[list(CORREDORES)[rng.integers(len(CORREDORES))]] if rng.random() < 0.7 else otras
        e = es[rng.choice(len(es), p=p)]
        L = e['lanes']; x = rng.random()
        if L == 1:
            carriles = [0]; sev = 'total'
        elif x < 0.75 or (L == 2 and x < 0.90):
            carriles = [int(rng.integers(L))]; sev = 'parcial'
        elif x < 0.90:
            l0 = int(rng.integers(L - 1)); carriles = [l0, l0 + 1]; sev = 'parcial'
        else:
            carriles = list(range(L)); sev = 'total'
        out.append({'inc_id': f'inc{dia_id:02d}_{k}', 'dia_id': dia_id, 'edge': e['id'], 'nombre_via': e['name'],
                    'corredor': e['corredor'] or 'ninguno', 'celda_id': e['celda'], 'uso_rutas': int(uso[e['id']]),
                    'pos': float(rng.uniform(0.35, 0.65) * e['len']), 'carriles': carriles, 'n_carriles_via': L,
                    'severidad': sev, 't_ini_plan': int(rng.uniform(t_min, t_max)),
                    'duracion': int(rng.uniform(600, 1500))})
    return out


# ----------------------------------------------------------------------------- simulación
class Colapso(RuntimeError):
    """La red quedó en gridlock (demasiados autos acumulados): la simulación se cortó."""


def correr_sumo(trabajo, semilla, t_fin, con_tripinfo=True, extra_rutas=(), log_file=None, extra=(), limite_en_red=None):
    """SUMO sin rerouting. Con limite_en_red, un vigilante lee el summary cada 30 s y corta la
    simulación (excepción Colapso) si los autos en la red superan ese límite."""
    rutas = ','.join(['/t/rutas.rou.xml.gz'] + list(extra_rutas))
    summary = 'summary.xml' if limite_en_red else 'summary.xml.gz'
    cmd = ['sumo', '-n', RED, '-r', rutas, '-a', '/t/vtypes.add.xml,/t/salidas.add.xml',
           '--begin', '0', '--end', str(t_fin), '--seed', str(semilla),
           '--no-step-log', '--no-warnings', '--time-to-teleport', '300', '--max-depart-delay', '900',
           # un auto detenido >30 s dentro de un cruce deja de bloquear a los transversales (evita gridlock
           # artificial de SUMO: los conductores reales se abren paso o no entran a la intersección)
           '--ignore-junction-blocker', '30',
           '--summary-output', f'/t/{summary}', '--stop-output', '/t/stops.xml',
           '--duration-log.statistics']
    if con_tripinfo:
        cmd += ['--tripinfo-output', '/t/tripinfo.xml.gz']
    cmd += list(extra)
    if not limite_en_red:
        docker(cmd, trabajo, log_file)
        return
    nombre = f'sumo_{Path(trabajo).name}_{os.getpid()}'
    full = ['docker', 'run', '--rm', '--name', nombre, '--ulimit', 'core=0', '-u', f'{os.getuid()}:{os.getgid()}',
            '-e', 'SUMO_HOME=/usr/share/sumo', '-v', f'{RAIZ}:/w:ro', '-v', f'{trabajo}:/t', '-w', '/w', IMG] + cmd
    f_sum = Path(trabajo) / summary
    with open(log_file or os.devnull, 'a') as f:
        pr = subprocess.Popen(full, stdout=f, stderr=subprocess.STDOUT)
        colapso = None
        while pr.poll() is None:
            time.sleep(30)
            try:
                with open(f_sum, 'rb') as fs:
                    fs.seek(max(0, f_sum.stat().st_size - 4000))
                    cola = fs.read().decode(errors='ignore')
                m = re.findall(r'time="([0-9.]+)"[^>]*running="([0-9]+)"', cola)
            except OSError:
                m = []
            if m and int(m[-1][1]) > limite_en_red:
                colapso = (float(m[-1][0]), int(m[-1][1]))
                subprocess.run(['docker', 'kill', nombre], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                pr.wait()
                break
    if colapso:
        raise Colapso(f'gridlock: {colapso[1]} autos en la red a t_sim={colapso[0]:.0f} s')
    if pr.returncode != 0:
        raise RuntimeError(f"falló sumo (código {pr.returncode}); ver {log_file}")
    with open(f_sum, 'rb') as fi, gzip.open(Path(trabajo) / 'summary.xml.gz', 'wb') as fo:
        shutil.copyfileobj(fi, fo)
    f_sum.unlink()


def duarouter(trabajo, semilla, log_file=None):
    """Rutas de menor tiempo en red libre con perturbación aleatoria de pesos (factor 1.5)
    para repartir rutas entre vías paralelas, como hacen los conductores reales."""
    docker(['duarouter', '-n', RED, '--route-files', '/t/viajes.trips.xml', '-o', '/t/rutas.rou.xml.gz',
            '--routing-algorithm', 'astar', '--weights.random-factor', '1.5', '--seed', str(semilla),
            '--ignore-errors', '--no-step-log', '--no-warnings', '--repair'], trabajo, log_file)
    for f in Path(trabajo).glob('rutas.rou*alt*'):      # alternativas: SUMO no las usa
        f.unlink()


# ----------------------------------------------------------------------------- lectura de salidas
def _iter(path, tag):
    with gzip.open(path) if str(path).endswith('.gz') else open(path, 'rb') as f:
        for ev, el in ET.iterparse(f, events=('end',)):
            if el.tag == tag:
                yield el
                el.clear()


def leer_summary(path):
    filas = []
    for s in _iter(path, 'step'):
        a = s.attrib
        filas.append((float(a['time']), int(a['running']), int(a['waiting']), int(a['arrived']), int(a['inserted']),
                      int(a.get('teleports', 0)), int(a.get('halting', 0)), float(a['meanSpeed']), float(a.get('meanSpeedRelative', 'nan'))))
    return pd.DataFrame(filas, columns=['t_sim', 'en_red', 'en_espera', 'llegados', 'insertados', 'teleports', 'detenidos', 'vel_media', 'vel_relativa'])


def leer_edgedata(path):
    filas = []
    for it in _iter(path, 'interval'):
        b = float(it.get('begin'))
        for e in it.iter('edge'):
            a = e.attrib
            filas.append((b, a['id'], float(a.get('sampledSeconds', 0)), float(a.get('speed', 'nan')), float(a.get('occupancy', 0)),
                          float(a.get('timeLoss', 0)), float(a.get('waitingTime', 0)), int(float(a.get('entered', 0)))))
    return pd.DataFrame(filas, columns=['t_sim', 'edge', 'ss', 'speed', 'occ', 'timeloss', 'wait', 'entered'])


def leer_e1(path):
    filas = []
    for i in _iter(path, 'interval'):
        a = i.attrib
        _, edge, lane = a['id'].split('|')
        filas.append((float(a['begin']), edge, int(lane), int(a['nVehContrib']), float(a['flow']), float(a['occupancy']),
                      float(a['speed']), float(a.get('harmonicMeanSpeed', a['speed']))))
    return pd.DataFrame(filas, columns=['t_sim', 'edge', 'lane', 'n', 'flujo', 'ocupacion', 'vel', 'vel_arm'])


def leer_tripinfo(path):
    filas = []
    for t in _iter(path, 'tripinfo'):
        a = t.attrib
        if a['vType'] == 'incidente':
            continue
        filas.append((a['id'], a['vType'] == 'auto_v2x', float(a['depart']), float(a['arrival']), float(a['duration']),
                      float(a['routeLength']), float(a['timeLoss']), float(a['waitingTime']), float(a['departDelay'])))
    return pd.DataFrame(filas, columns=['veh_id', 'v2x', 'depart', 'arrival', 'duracion', 'longitud', 'perdida', 'espera', 'retraso_salida'])


def leer_stops(path):
    filas = []
    if not Path(path).exists():
        return pd.DataFrame(columns=['veh', 'inicio', 'fin'])
    for s in _iter(path, 'stopinfo'):
        filas.append((s.get('id'), float(s.get('started')), float(s.get('ended', 'nan'))))
    return pd.DataFrame(filas, columns=['veh', 'inicio', 'fin'])


# ----------------------------------------------------------------------------- agregación
def agregar_celdas(ed, periodo=300):
    """edgeData por arista -> métricas por celda e intervalo.
    n_veh: vehículos promedio presentes; densidad: veh/km/carril sobre TODA la longitud de
    carril de la celda; vel: velocidad espacial media (distancia total / tiempo total, km/h);
    flujo = densidad x vel (veh/h/carril); ocupacion: % ponderado por longitud de carril;
    perdida/espera: segundos acumulados de pérdida de tiempo y espera."""
    ar = tabla_aristas().set_index('id')
    ar['lane_km'] = ar['len'] * ar['lanes'] / 1000
    km_celda = ar.groupby('celda')['lane_km'].sum()
    ed = ed.join(ar[['celda', 'lane_km']], on='edge', how='inner')
    ed['dist'] = ed['speed'] * ed['ss']
    ed['occ_w'] = ed['occ'] * ed['lane_km']
    g = ed.groupby(['t_sim', 'celda']).agg(ss=('ss', 'sum'), dist=('dist', 'sum'), occ_w=('occ_w', 'sum'),
                                           perdida=('timeloss', 'sum'), espera=('wait', 'sum')).reset_index()
    g['n_veh'] = g['ss'] / periodo
    km = g['celda'].map(km_celda)
    g['densidad'] = g['n_veh'] / km
    g['vel'] = np.where(g['ss'] > 0, g['dist'] / g['ss'] * 3.6, np.nan)
    g['flujo'] = g['densidad'] * g['vel']
    g['ocupacion'] = g['occ_w'] / km
    return g[['t_sim', 'celda', 'n_veh', 'densidad', 'vel', 'flujo', 'ocupacion', 'perdida', 'espera']]


def tabla_celdas():
    """Celdas activas (con al menos una arista de autos) y su velocidad libre ponderada."""
    r = red_info(); ar = tabla_aristas()
    ar['lk'] = ar['len'] * ar['lanes']
    c = ar.groupby('celda').apply(lambda d: pd.Series({
        'v_ff': float((d['speed'] * 3.6 * d['lk']).sum() / d['lk'].sum()),
        'lane_km': float(d['lk'].sum() / 1000), 'n_aristas': len(d),
        'corredor': ','.join(sorted(set(d['corredor'].dropna())))}), include_groups=False).reset_index()
    c['fila'] = c['celda'] // r['nx']; c['col'] = c['celda'] % r['nx']
    return c
