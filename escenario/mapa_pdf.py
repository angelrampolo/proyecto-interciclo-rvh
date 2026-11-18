#!/usr/bin/env python3
"""Genera el PDF del mapa de la red de Cuenca urbana (solo autos) para revisión visual.
Páginas: 1) jerarquía vial + semáforos + parroquias, 2) límites de velocidad,
3) sentidos de circulación (zoom al Centro Histórico), 4) malla 200 m y corredores AID.
Uso: ~/venv_cpu/bin/python escenario/mapa_pdf.py   -> escenario/mapa_cuenca_urbana.pdf"""
import json, math, sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / 'etapa2'))
import comun as C

R = C.red_info()
E = R['edges']
OFF = R['offset']


def utm17s(lat, lon):
    """Lat/lon WGS84 -> UTM zona 17 sur (fórmulas estándar de Snyder)."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f); ep2 = e2 / (1 - e2); k0 = 0.9996
    lon0 = math.radians(-81)
    phi, lam = math.radians(lat), math.radians(lon)
    N = a / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    T = math.tan(phi) ** 2; Cc = ep2 * math.cos(phi) ** 2; A = math.cos(phi) * (lam - lon0)
    M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * phi)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * phi) - (35 * e2 ** 3 / 3072) * math.sin(6 * phi))
    x = k0 * N * (A + (1 - T + Cc) * A ** 3 / 6 + (5 - 18 * T + T ** 2 + 72 * Cc - 58 * ep2) * A ** 5 / 120) + 500000
    y = k0 * (M + N * math.tan(phi) * (A ** 2 / 2 + (5 - T + 9 * Cc + 4 * Cc ** 2) * A ** 4 / 24
                                       + (61 - 58 * T + T ** 2 + 600 * Cc - 330 * ep2) * A ** 6 / 720))  # proj de la red sin +south: sin falso norte
    return x + OFF[0], y + OFF[1]


def parroquias():
    d = json.load(open(RAIZ / 'osm/parroquias_geom.json'))
    nombres = {3842960: 'Hermano Miguel', 3842961: 'Machángara'}
    out = []
    for rel in d['elements']:
        nombre = rel['tags'].get('name', nombres.get(rel['id'], '?'))
        segs = [[utm17s(p['lat'], p['lon']) for p in m['geometry']] for m in rel['members']
                if m['type'] == 'way' and m.get('role') in ('outer', '') and 'geometry' in m]
        out.append((nombre, segs))
    return out


JER = [  # (tipos, color, ancho, etiqueta)
    (('highway.motorway', 'highway.trunk', 'highway.motorway_link', 'highway.trunk_link'), '#7f1d1d', 2.2, 'Vía rápida / troncal'),
    (('highway.primary', 'highway.primary_link'), '#dc2626', 1.6, 'Primaria (avenidas)'),
    (('highway.secondary', 'highway.secondary_link'), '#f97316', 1.1, 'Secundaria'),
    (('highway.tertiary', 'highway.tertiary_link'), '#ca8a04', 0.8, 'Terciaria'),
    (('highway.residential', 'highway.unclassified'), '#9ca3af', 0.35, 'Residencial / local'),
]


def base(ax, titulo, alpha=1.0):
    b = R['boundary']
    ax.set_xlim(b[0] - 300, b[2] + 300); ax.set_ylim(b[1] - 300, b[3] + 300)
    ax.set_aspect('equal'); ax.set_title(titulo, fontsize=15, fontweight='bold', loc='left')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#d1d5db')
    # barra de escala 2 km
    x0, y0 = b[0] + 200, b[1] + 100
    ax.plot([x0, x0 + 2000], [y0, y0], color='k', lw=3); ax.text(x0 + 1000, y0 + 150, '2 km', ha='center', fontsize=9)
    ax.annotate('N', xy=(b[2] - 100, b[3] - 200), xytext=(b[2] - 100, b[3] - 1100), ha='center', fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='-|>', color='k', lw=1.5))


def dibujar_parroquias(ax, etiquetas=True):
    for nombre, segs in parroquias():
        for s in segs:
            xs, ys = zip(*s); ax.plot(xs, ys, color='#6b7280', lw=0.6, ls='--', alpha=.7, zorder=1)
        if etiquetas and segs:
            pts = [p for s in segs for p in s]
            cx = sum(p[0] for p in pts) / len(pts); cy = sum(p[1] for p in pts) / len(pts)
            ax.text(cx, cy, nombre, fontsize=6.5, color='#374151', ha='center', alpha=.85, style='italic', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=.6))


def lineas(ax, edges, color, lw, z=2, alpha=1.0):
    ax.add_collection(LineCollection([e['shape'] for e in edges], colors=color, linewidths=lw, zorder=z, alpha=alpha, capstyle='round'))


def pagina_jerarquia(pdf):
    fig, ax = plt.subplots(figsize=(16.5, 11.7))
    base(ax, 'Red vial de Cuenca urbana para SUMO — solo autos (15 parroquias urbanas)')
    dibujar_parroquias(ax)
    for tipos, col, lw, _ in reversed(JER):
        lineas(ax, [e for e in E if e['type'] in tipos], col, lw)
    tx, ty = zip(*R['tls']); ax.scatter(tx, ty, s=9, c='#111827', zorder=6, label='Semáforo')
    h = [Line2D([], [], color=c, lw=max(l * 2, 1.5), label=t) for _, c, l, t in JER]
    h.append(Line2D([], [], marker='o', ls='', color='#111827', ms=4, label=f'Semáforo ({len(R["tls"])})'))
    h.append(Line2D([], [], color='#6b7280', ls='--', lw=1, label='Límite parroquial'))
    ax.legend(handles=h, loc='upper left', fontsize=10, framealpha=.95)
    km = sum(e['len'] for e in E) / 1000
    txt = (f"Aristas: {len(E):,}\nLongitud de vías: {km:,.0f} km\nKm-carril: {sum(e['len'] * e['lanes'] for e in E) / 1000:,.0f}\n"
           f"Semáforos (actuados): {len(R['tls'])}\nExtensión: {(R['boundary'][2] - R['boundary'][0]) / 1000:.1f} × {(R['boundary'][3] - R['boundary'][1]) / 1000:.1f} km\n"
           f"Fuente: OpenStreetMap (nov 2026)\nPoda: calles sin salida y\naristas no conexas eliminadas")
    ax.text(0.985, 0.03, txt, transform=ax.transAxes, ha='right', va='bottom', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', fc='white', ec='#9ca3af'))
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def pagina_velocidad(pdf):
    fig, ax = plt.subplots(figsize=(16.5, 11.7))
    base(ax, 'Límites de velocidad asignados (OSM maxspeed; si falta: 50 km/h urbano, 70–80 km/h vías rápidas)')
    dibujar_parroquias(ax, etiquetas=False)
    grupos = [(0, 31, '#16a34a', 0.4, '≤ 30 km/h'), (31, 45, '#84cc16', 0.6, '31–45 km/h'), (45, 55, '#f59e0b', 0.8, '50 km/h'),
              (55, 75, '#dc2626', 1.4, '60–70 km/h'), (75, 200, '#7f1d1d', 2.0, '≥ 80 km/h')]
    h = []
    for lo, hi, c, lw, t in grupos:
        sel = [e for e in E if lo <= e['speed'] * 3.6 < hi]
        lineas(ax, sel, c, lw)
        h.append(Line2D([], [], color=c, lw=3, label=f"{t}  ({sum(e['len'] for e in sel) / 1000:,.0f} km)"))
    ax.legend(handles=h, loc='upper left', fontsize=11, title='Límite de velocidad', framealpha=.95)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def pagina_sentidos(pdf):
    pares = {(e['from'], e['to']) for e in E}
    unico = [e for e in E if (e['to'], e['from']) not in pares]
    doble = [e for e in E if (e['to'], e['from']) in pares]
    fig, axs = plt.subplots(1, 2, figsize=(16.5, 11.7), gridspec_kw={'width_ratios': [1.35, 1]})
    ax = axs[0]
    base(ax, 'Sentidos de circulación')
    lineas(ax, doble, '#9ca3af', 0.35); lineas(ax, unico, '#2563eb', 0.6)
    # recuadro del Centro Histórico (El Sagrario / Gil Ramírez Dávalos / San Blas)
    cx0, cy0 = utm17s(-2.9045, -79.0115); cx1, cy1 = utm17s(-2.8930, -78.9960)
    ax.add_patch(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, fill=False, ec='k', lw=1.5, zorder=8))
    ax.legend(handles=[Line2D([], [], color='#2563eb', lw=3, label=f'Sentido único ({len(unico):,} aristas)'),
                       Line2D([], [], color='#9ca3af', lw=3, label=f'Doble sentido ({len(doble):,} aristas)')],
              loc='upper left', fontsize=11, framealpha=.95)
    ax = axs[1]
    ax.set_title('Zoom: Centro Histórico (flechas = sentido)', fontsize=13, fontweight='bold', loc='left')
    ax.set_xlim(cx0, cx1); ax.set_ylim(cy0, cy1); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    sel = [e for e in E if any(cx0 - 50 <= x <= cx1 + 50 and cy0 - 50 <= y <= cy1 + 50 for x, y in e['shape'])]
    for e in sel:
        u = (e['to'], e['from']) not in pares
        xs, ys = zip(*e['shape'])
        ax.plot(xs, ys, color='#2563eb' if u else '#6b7280', lw=1.2 + 0.8 * (e['lanes'] - 1), solid_capstyle='round', zorder=2)
        if e['len'] > 40:
            sh = e['shape']; k = max(len(sh) // 2 - 1, 0); (x0, y0), (x1, y1) = sh[k], sh[k + 1]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2; d = math.hypot(x1 - x0, y1 - y0) or 1
            if not u:  # desplazar a la derecha para separar ambos sentidos
                mx += (y1 - y0) / d * 6; my -= (x1 - x0) / d * 6
            ax.annotate('', xy=(mx + (x1 - x0) / d * 10, my + (y1 - y0) / d * 10), xytext=(mx - (x1 - x0) / d * 10, my - (y1 - y0) / d * 10),
                        arrowprops=dict(arrowstyle='-|>', color='#1e3a8a' if u else '#374151', lw=0.9, mutation_scale=9), zorder=4)
    tx = [(x, y) for x, y in R['tls'] if cx0 <= x <= cx1 and cy0 <= y <= cy1]
    if tx:
        ax.scatter(*zip(*tx), s=40, c='#dc2626', edgecolor='k', zorder=6)
    ax.plot([cx0 + 50, cx0 + 250], [cy0 + 40, cy0 + 40], 'k', lw=3); ax.text(cx0 + 150, cy0 + 60, '200 m', ha='center', fontsize=9)
    ax.legend(handles=[Line2D([], [], color='#2563eb', lw=3, label='Sentido único'), Line2D([], [], color='#6b7280', lw=3, label='Doble sentido'),
                       Line2D([], [], marker='o', ls='', mfc='#dc2626', mec='k', ms=7, label='Semáforo')], loc='upper right', fontsize=10, framealpha=.95)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def pagina_malla(pdf):
    celdas = C.tabla_celdas(); b = R['boundary']
    fig, ax = plt.subplots(figsize=(16.5, 11.7))
    base(ax, f"Malla espacial de {C.CELDA:.0f} m ({R['nx']} × {R['ny']} celdas, {len(celdas):,} activas) y corredores AID")
    rects = [Rectangle((b[0] + c * C.CELDA, b[1] + f * C.CELDA), C.CELDA, C.CELDA) for f, c in zip(celdas.fila, celdas.col)]
    ax.add_collection(PatchCollection(rects, facecolor='#dbeafe', edgecolor='#93c5fd', linewidths=0.2, zorder=0))
    lineas(ax, E, '#9ca3af', 0.3)
    cols = {'C1_Americas': '#dc2626', 'C2_Circunvalacion_Sur': '#7c3aed', 'C3_RemigioCrespo_Solano': '#059669'}
    h = [Rectangle((0, 0), 1, 1, fc='#dbeafe', ec='#93c5fd', label='Celda activa (tiene vías)')]
    for c, col in cols.items():
        sel = [e for e in E if e['corredor'] == c]
        lineas(ax, sel, col, 2.6, z=4)
        h.append(Line2D([], [], color=col, lw=4, label=f"{c.replace('_', ' ')}  ({sum(e['len'] for e in sel) / 1000:.1f} km, {len({e['celda'] for e in sel})} celdas)"))
    ax.legend(handles=h, loc='upper left', fontsize=11, framealpha=.95)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def main():
    salida = RAIZ / 'escenario/mapa_cuenca_urbana.pdf'
    with PdfPages(salida) as pdf:
        pagina_jerarquia(pdf); pagina_velocidad(pdf); pagina_sentidos(pdf); pagina_malla(pdf)
        d = pdf.infodict(); d['Title'] = 'Red vial de Cuenca urbana (SUMO) - Proyecto Interciclo RVH'
    print(salida)


if __name__ == '__main__':
    main()
