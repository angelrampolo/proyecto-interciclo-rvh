"""Prediccion de la velocidad de toda Cuenca a +15/+30/+45/+60 min con un GCN-GRU sobre el grafo vial.

Idea: el promedio historico (misma hora, mismo tipo de dia) ya explica la parte recurrente del trafico.
La red aprende la CORRECCION sobre ese promedio a partir de la ultima hora observada y de los vecinos
en el grafo de calles (aguas arriba y aguas abajo). Asi el modelo nunca puede quedar peor que el
promedio historico por falta de datos y se concentra en lo que el promedio no ve (incidentes, lluvia,
dias mas cargados).

Uso:  python entrenar_prediccion.py --modo gt     (sensores, 0 % V2X)
      python entrenar_prediccion.py --modo obs    (solo autos conectados V2X)
"""
import argparse, json, time
import numpy as np, pandas as pd, torch, torch.nn as nn
import comun as C

torch.set_num_threads(int(__import__("os").environ.get("HILOS", "8")))


class GCNGRU(nn.Module):
    def __init__(self, N, ij, cin=6, h=32, ntiempo=6, nh=len(C.HORIZONTES)):
        super().__init__()
        self.N = N
        self.A_in = self._norm(ij[1], ij[0], N)    # agrega desde las celdas de aguas arriba
        self.A_out = self._norm(ij[0], ij[1], N)   # agrega desde las celdas de aguas abajo
        self.inp = nn.Linear(cin, h)
        self.g1 = nn.Linear(3 * h, h)
        self.gru = nn.GRU(h, h, batch_first=True)
        self.g2 = nn.Linear(3 * h, h)
        self.emb = nn.Embedding(N, 8)
        self.head = nn.Sequential(nn.Linear(h + 8 + ntiempo + nh + 1, 64), nn.ReLU(), nn.Linear(64, nh))

    @staticmethod
    def _norm(fila, col, N):
        grado = np.bincount(fila, minlength=N).astype(np.float32)
        w = 1.0 / np.maximum(grado[fila], 1)
        return torch.sparse_coo_tensor(np.stack([fila, col]), w, (N, N)).coalesce()

    def _gcn(self, x, capa):              # x [S, N, h]
        S, N, h = x.shape
        flat = x.permute(1, 0, 2).reshape(N, S * h)
        a = torch.sparse.mm(self.A_in, flat).reshape(N, S, h).permute(1, 0, 2)
        b = torch.sparse.mm(self.A_out, flat).reshape(N, S, h).permute(1, 0, 2)
        return torch.relu(capa(torch.cat([x, a, b], -1)))

    def forward(self, x, tiempo, ha_rel):
        # x [B,12,N,6]; tiempo [B,6]; ha_rel [B,N,4] -> correccion relativa [B,N,4]
        B, T, N, _ = x.shape
        z = torch.relu(self.inp(x)).reshape(B * T, N, -1)
        z = self._gcn(z, self.g1).reshape(B, T, N, -1).permute(0, 2, 1, 3).reshape(B * N, T, -1)
        _, hN = self.gru(z)
        z = self._gcn(hN[-1].reshape(B, N, -1), self.g2)
        e = self.emb.weight[None].expand(B, -1, -1)
        tt = tiempo[:, None, :].expand(-1, N, -1)
        ultimo = x[:, -1, :, 0:1]
        return self.head(torch.cat([z, e, tt, ha_rel, ultimo], -1))


class Datos:
    """Ventanas armadas al vuelo desde los tensores por dia (no se copian 2 GB de ventanas)."""
    def __init__(self, lista, modo, HA=None):
        X, self.Y, self.M, self.meta = C.cargar_dias(lista, modo)
        self.vff = C.celdas().v_ff.to_numpy(np.float32)
        if HA is None:
            HA = C.promedio_historico(self.Y, self.meta)
        self.HA = HA
        self.tipo = np.array([C.TIPOS.index(t) for t in self.meta.tipo_dia])
        ha_rel = HA / self.vff                                        # [3,288,N]
        self.ha_rel = ha_rel
        des = X[..., 0] - ha_rel[self.tipo]                           # desvio respecto a lo habitual
        self.X = np.concatenate([X, des[..., None]], -1)
        self.idx = C.indice_ventanas(len(lista))
        self.tiempo = np.stack([C.rasgos_tiempo(t + C.H, self.meta.iloc[d]) for d, t in self.idx])

    def lote(self, sel):
        x = np.stack([self.X[d, t - C.HIST + 1:t + 1] for d, t in (self.idx[i] for i in sel)])
        y = np.stack([[self.Y[d, t + k] for k in C.HORIZONTES] for d, t in (self.idx[i] for i in sel)])
        m = np.stack([[self.M[d, t + k] for k in C.HORIZONTES] for d, t in (self.idx[i] for i in sel)])
        ha = np.stack([[self.ha_rel[self.tipo[d], t + k] for k in C.HORIZONTES] for d, t in (self.idx[i] for i in sel)])
        f = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        return f(x), f(self.tiempo[sel]), f(ha.transpose(0, 2, 1)), f(y.transpose(0, 2, 1)), f(m.transpose(0, 2, 1))


def normalizar(datos, mu=None, sd=None):
    cols = [1, 2, 3]            # ocupacion, densidad, flujo; vel_rel, hay_dato y desvio ya estan en escala ~1
    if mu is None:
        mu = datos.X[..., cols].reshape(-1, 3).mean(0)
        sd = datos.X[..., cols].reshape(-1, 3).std(0) + 1e-6
    datos.X[..., cols] = (datos.X[..., cols] - mu) / sd
    return mu, sd


def predecir(modelo, datos, bs=24):
    modelo.eval()
    P = []
    with torch.no_grad():
        for i in range(0, len(datos.idx), bs):
            x, tt, ha, y, m = datos.lote(range(i, min(i + bs, len(datos.idx))))
            P.append(((ha + modelo(x, tt, ha)) * torch.from_numpy(datos.vff)[None, :, None]).numpy())
    return np.concatenate(P)          # [ventanas, N, 4] km/h


def objetivos(datos):
    Y = np.stack([[datos.Y[d, t + k] for k in C.HORIZONTES] for d, t in datos.idx]).transpose(0, 2, 1)
    M = np.stack([[datos.M[d, t + k] for k in C.HORIZONTES] for d, t in datos.idx]).transpose(0, 2, 1)
    return Y, M


def bases(datos):
    pers = np.stack([np.repeat(datos.Y[d, t][:, None], len(C.HORIZONTES), 1) for d, t in datos.idx])
    ha = np.stack([[datos.HA[datos.tipo[d], t + k] for k in C.HORIZONTES] for d, t in datos.idx]).transpose(0, 2, 1)
    return pers, ha


def evaluar(nombre, datos, P, extra=None):
    Y, M = objetivos(datos)
    pers, ha = bases(datos)
    cel = C.celdas()
    corredor = (cel.corredor.fillna("ninguno") != "ninguno").to_numpy()
    pico = np.array([C.es_pico(t + C.H) for d, t in datos.idx])
    inc = np.zeros_like(M[..., -1])
    tabla_inc = pd.read_parquet(C.DS / "incidentes.parquet")
    pos = {c: i for i, c in enumerate(cel.celda)}
    dias_ids = datos.meta.dia_id.tolist()
    for _, r in tabla_inc[tabla_inc.dia_id.isin(dias_ids)].iterrows():
        dd = dias_ids.index(r.dia_id)
        for w, (d, t) in enumerate(datos.idx):
            tobj = (t + C.H) * 300
            if d == dd and r.t_ini - 600 <= tobj <= r.t_fin + 1800 and r.celda_id in pos:
                inc[w, pos[r.celda_id]] = True
    res = {}
    for etq, PP in (("persistencia", pers), ("promedio_historico", ha), ("modelo", P)):
        r = {}
        for j, k in enumerate(C.HORIZONTES):
            r[f"+{k*5}min"] = C.metricas(Y[..., j], PP[..., j], M[..., j])
        Y60, P60, M60 = Y[..., -1], PP[..., -1], M[..., -1]
        r["desglose_60min"] = {
            "hora_pico": C.metricas(Y60[pico], P60[pico], M60[pico]),
            "valle": C.metricas(Y60[~pico], P60[~pico], M60[~pico]),
            "corredores": C.metricas(Y60, P60, M60 & corredor[None]),
            "resto_de_celdas": C.metricas(Y60, P60, M60 & ~corredor[None]),
            "celdas_con_incidente": C.metricas(Y60, P60, M60 & inc) if (M60 & inc).any() else None,
            "congestion_real(v<0.55vff)": C.metricas(Y60, P60, M60 & (Y60 < 0.55 * datos.vff[None])),
        }
        for tp in C.TIPOS:
            sel = np.array([datos.meta.tipo_dia.iloc[d] == tp for d, t in datos.idx])
            if sel.any():
                r["desglose_60min"][f"dia_{tp}"] = C.metricas(Y60[sel], P60[sel], M60[sel])
        res[etq] = r
    if extra:
        res.update(extra)
    print(f"\n== {nombre} (+60 min, celdas con autos) ==")
    for etq in ("persistencia", "promedio_historico", "modelo"):
        print(f"  {etq:20s}", res[etq]["+60min"], "| pico", res[etq]["desglose_60min"]["hora_pico"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["gt", "obs"], default="gt")
    ap.add_argument("--epocas", type=int, default=12)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    t0 = time.time()
    tr = Datos(C.ids("train"), a.modo)
    mu, sd = normalizar(tr)
    va = Datos(C.ids("val"), a.modo, tr.HA); normalizar(va, mu, sd)
    print(f"[{a.modo}] train {len(tr.idx)} ventanas, val {len(va.idx)} | carga {time.time()-t0:.0f}s", flush=True)
    ij = C.adyacencia()
    modelo = GCNGRU(len(tr.vff), ij)
    opt = torch.optim.AdamW(modelo.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epocas)
    peso_h = torch.tensor([0.6, 0.8, 1.0, 1.6])
    vff_t = torch.from_numpy(tr.vff)[None, :, None]
    Yva, Mva = objetivos(va)
    hist, mejor, estado = [], 1e9, None
    for ep in range(1, a.epocas + 1):
        modelo.train(); te = time.time(); perd = []
        orden = np.random.permutation(len(tr.idx))
        for i in range(0, len(orden), a.bs):
            x, tt, ha, y, m = tr.lote(orden[i:i + a.bs])
            p = (ha + modelo(x, tt, ha)) * vff_t
            w = m.float() * peso_h * (1 + 1.5 * (y < 0.55 * vff_t).float())   # pesa mas la congestion
            l = (nn.functional.smooth_l1_loss(p, y, reduction="none", beta=2.0) * w).sum() / w.sum()
            opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(modelo.parameters(), 1.0); opt.step()
            perd.append(l.item())
        sched.step()
        P = predecir(modelo, va)
        mv = C.metricas(Yva[..., -1], P[..., -1], Mva[..., -1])
        hist.append({"epoca": ep, "perdida_train": round(float(np.mean(perd)), 4), **{f"val_{k}": v for k, v in mv.items() if k != "n"},
                     "seg": round(time.time() - te)})
        print(hist[-1], flush=True)
        if mv["MAE"] < mejor:
            mejor, estado = mv["MAE"], {k: v.clone() for k, v in modelo.state_dict().items()}
    modelo.load_state_dict(estado)
    torch.save({"estado": estado, "mu": mu, "sd": sd, "HA": tr.HA, "modo": a.modo, "ij": ij},
               C.SALIDA / f"modelo_gcngru_{a.modo}.pt")
    del tr
    res = {"modo": a.modo, "historial": hist, "parametros": sum(p.numel() for p in modelo.parameters())}
    res["validacion"] = evaluar("VAL", va, predecir(modelo, va)); del va
    te_ = Datos(C.ids("test"), a.modo, tr_HA := torch.load(C.SALIDA / f"modelo_gcngru_{a.modo}.pt", weights_only=False)["HA"])
    normalizar(te_, mu, sd)
    Pte = predecir(modelo, te_)
    res["test"] = evaluar("TEST (una sola vez)", te_, Pte)
    # predicciones del test para el dashboard (km/h enteros)
    np.savez_compressed(C.SALIDA / f"pred_test_{a.modo}.npz", P=np.clip(Pte, 0, 120).astype(np.uint8),
                        idx=np.array(te_.idx), dias=te_.meta.dia_id.to_numpy(),
                        Y=np.clip(te_.Y, 0, 120).astype(np.uint8), HA=np.clip(tr_HA, 0, 120).astype(np.uint8))
    del te_
    res["estres"] = {}
    for d in C.ids("test_estres"):
        ds = Datos([d], a.modo, tr_HA); normalizar(ds, mu, sd)
        r = evaluar(f"ESTRES dia {d}", ds, predecir(modelo, ds))
        res["estres"][f"dia_{d}"] = {k: r[k]["+60min"] for k in ("persistencia", "promedio_historico", "modelo")}
    json.dump(res, open(C.SALIDA / f"metricas_prediccion_{a.modo}.json", "w"), indent=2, ensure_ascii=False)
    print(f"listo en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
