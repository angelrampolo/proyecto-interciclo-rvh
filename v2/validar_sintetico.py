"""Validacion del dato sintetico frente a cifras reales publicadas de Cuenca (anexo C del informe).

Compara magnitudes del dataset (SUMO) que tienen una contraparte real citada:
  1. volumen de la Av. de las Americas en hora pico y por dia (seccion completa, ambos sentidos, central y laterales);
  2. horas pico de la demanda;
  3. viajes en auto por dia;
  4. efecto de la lluvia sobre la velocidad;
  5. limites de velocidad de la Av. de las Americas.
Guarda resultados/validacion_sintetico.json. Uso: python v2/validar_sintetico.py
"""
import json
import numpy as np
import pandas as pd
import comun as C

# cifras reales con su fuente (claves de informe/referencias.bib)
REAL = {
    "americas_pico_veh_h": {"valor": [4500, 5400], "fuente": "elmercurio2021americas, elmercurio2022americas",
                            "nota": "4500 veh/h un jueves entre el Parque Jacaranda y la gasolinera del Sindicato de Choferes; ~5400 veh/h en horas pico en el sector del Parque Industrial"},
    "americas_dia_veh": {"valor": 75000, "fuente": "elmercurio2021americas"},
    "horas_pico": {"valor": ["07:00-08:00", "18:00-19:00"], "fuente": "jacome2024radar", "nota": "radares en la autopista Cuenca-Azogues; pico de mediodia en pena2021canaris"},
    "viajes_auto_dia": {"valor": 400000, "fuente": "elmercurio2026movilidad, cazorla2026unimobility, elmercurio2020carros",
                        "nota": "1,5 M viajes/dia x 32 % en auto / 1,2 personas por auto"},
    "lluvia_vel_libre_pct": {"valor": [-2, -9], "fuente": "hranac2006weather"},
    "lluvia_capacidad_pct": {"valor": [-10, -11], "fuente": "hranac2006weather"},
    "americas_limites_kmh": {"valor": [50, 70], "fuente": "canafe2026irap", "nota": "50 km/h via lateral, 70 km/h via central"},
}


def cruza(p, q, a, b):
    """True si el segmento pq corta al segmento ab."""
    def o(u, v, w):
        return np.sign((v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0]))
    return o(p, q, a) != o(p, q, b) and o(a, b, p) != o(a, b, q)


def secciones_americas():
    """Para cada celda de 200 m de la Av. de las Americas, las aristas que cruzan una linea transversal por su centro.
    Sumar esas aristas da el volumen de la seccion completa (ambos sentidos, via central y laterales), como un aforo."""
    red = json.load(open(C.RED))
    ari = pd.read_parquet(C.DS / "aristas.parquet")
    c1 = ari[ari.corredor == "C1_Americas"]
    forma = {e["id"]: np.array(e["shape"]) for e in red["edges"] if e["id"] in set(c1.id)}
    cel = C.celdas().set_index("celda")
    x0, y0 = red["boundary"][0], red["boundary"][1]
    out = {}
    for celda, g in c1.groupby("celda"):
        pts = np.vstack([forma[e] for e in g.id if e in forma])
        if len(pts) < 3:
            continue
        cx, cy = pts.mean(0)
        u = np.linalg.svd(pts - pts.mean(0))[2][0]         # eje principal de la avenida en la celda
        n = np.array([-u[1], u[0]]) * 120                  # linea transversal de 240 m
        a, b = np.array([cx, cy]) - n, np.array([cx, cy]) + n
        ids = [e for e in g.id if e in forma and any(cruza(p, q, a, b) for p, q in zip(forma[e][:-1], forma[e][1:]))]
        if len(ids) >= 2:   # una seccion real tiene al menos los dos sentidos
            out[int(celda)] = ids
    return out


def main():
    dias = C.dias()
    lab = dias[(dias.tipo_dia == "laborable") & (~dias.lluvia.astype(bool)) & (dias.split.isin(["train", "val", "test"]))]
    sec = secciones_americas()
    todas = sorted({e for v in sec.values() for e in v})
    pico, diario = {c: [] for c in sec}, {c: [] for c in sec}
    for d in lab.index:
        df = pd.read_parquet(C.DS / "corredor_1min" / f"dia_{d:02d}.parquet", columns=["t", "edge", "gt_n"],
                             filters=[("edge", "in", todas)])
        df["h"] = df.t // 3600
        hora = df.groupby(["edge", "h"]).gt_n.sum()
        for c, ids in sec.items():
            s = hora[hora.index.get_level_values(0).isin(ids)].groupby(level=1).sum()
            pico[c].append(float(s.max())); diario[c].append(float(s.sum()))
    p = np.array([np.mean(v) for v in pico.values()]); dd = np.array([np.mean(v) for v in diario.values()])
    # horas pico: autos en la red por hora, laborables secos
    r = pd.read_parquet(C.DS / "red_5min")
    r = r[r.dia_id.isin(lab.index)]
    perfil = r.assign(h=r.t // 3600).groupby("h").en_red.mean()
    manana, tarde = int(perfil.loc[5:10].idxmax()), int(perfil.loc[15:21].idxmax())
    lluvia = dias[dias.lluvia.astype(bool) & dias.split.isin(["train", "val", "test"])]
    sint = {
        "americas_pico_veh_h": {"mediana_secciones": round(float(np.median(p))), "p90": round(float(np.percentile(p, 90))),
                                "max": round(float(p.max())), "secciones": len(p)},
        "americas_dia_veh": {"mediana_secciones": round(float(np.median(dd))), "p90": round(float(np.percentile(dd, 90))), "max": round(float(dd.max()))},
        "horas_pico": {"manana": f"{manana:02d}:00-{manana + 1:02d}:00", "tarde": f"{tarde:02d}:00-{tarde + 1:02d}:00",
                       "mediodia_relativo": round(float(perfil.loc[12:14].max() / perfil.loc[5:10].max()), 2)},
        "viajes_auto_dia": {"laborable_seco": round(float(lab.n_viajes_completados.mean()))},
        "lluvia_vel_viaje_pct": round(float(100 * (lluvia.vel_media_viaje_kmh.mean() / lab.vel_media_viaje_kmh.mean() - 1)), 1),
        "lluvia_parametros": {"speedFactor": 0.88, "demanda_pct": 4},
        "americas_limites_kmh": sorted({int(round(v * 3.6)) for v in pd.read_parquet(C.DS / "aristas.parquet").query("corredor == 'C1_Americas'").speed}),
        "dias_usados": [int(x) for x in lab.index],
    }
    res = {"real": REAL, "sintetico": sint,
           "nota": "Datos sinteticos de SUMO. Laborables secos de train, val y test. Volumen de seccion = autos que cruzan una linea transversal por el centro de cada celda de 200 m (ambos sentidos, central y laterales)."}
    json.dump(res, open(C.SALIDA / "validacion_sintetico.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(sint, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
