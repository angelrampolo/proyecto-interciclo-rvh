#!/usr/bin/env python3
"""Poda iterativa de calles sin salida (dead-ends), igual que el script de las
prácticas (results/cuenca/prune_deadends.py), pero sin podar vías férreas
(tranvía: sus terminales son extremos legítimos).
Uso: python3 prune_deadends.py red.net.xml prune.txt"""
import os, sys
sys.path.append(os.path.join(os.environ.get('SUMO_HOME', '/usr/share/sumo'), 'tools'))
import sumolib

net = sumolib.net.readNet(sys.argv[1])
es_riel = lambda e: e.getType().startswith('railway')
removed, changed, vueltas = set(), True, 0
while changed:
    changed, vueltas = False, vueltas + 1
    for node in net.getNodes():
        out_e = [e for e in node.getOutgoing() if e.getID() not in removed and not es_riel(e)]
        in_e = [e for e in node.getIncoming() if e.getID() not in removed and not es_riel(e)]
        if any(es_riel(e) for e in node.getOutgoing() + node.getIncoming()):
            continue
        calles = {e.getID().lstrip('-') for e in out_e + in_e}
        if len(calles) == 1:
            removed.update(e.getID() for e in out_e + in_e)
            changed = True
print(f"Calles sin salida a podar: {len(removed)} (en {vueltas} pasadas)")

# Segunda limpieza: aristas de autos fuera de la mayor componente fuertemente
# conexa (se puede entrar pero no salir, o al revés) -> rutas imposibles.
import collections
car = [e for e in net.getEdges() if e.allows('passenger') and e.getID() not in removed and not es_riel(e)]
S = set(car)
succ = {e: [n for n in e.getOutgoing() if n in S] for e in car}
pred = collections.defaultdict(list)
for e, ns in succ.items():
    for n in ns:
        pred[n].append(e)
seen, order = set(), []
for s in car:
    if s in seen:
        continue
    seen.add(s); stack = [(s, iter(succ[s]))]
    while stack:
        v, it = stack[-1]
        for w in it:
            if w not in seen:
                seen.add(w); stack.append((w, iter(succ[w]))); break
        else:
            order.append(v); stack.pop()
comp, mejor = {}, []
for s in reversed(order):
    if s in comp:
        continue
    c, i = [s], 0; comp[s] = s
    while i < len(c):
        for w in pred[c[i]]:
            if w not in comp:
                comp[w] = s; c.append(w)
        i += 1
    if len(c) > len(mejor):
        mejor = c
fuera = {e.getID() for e in car} - {e.getID() for e in mejor}
removed |= fuera
print(f"Aristas fuera de la componente conexa principal: {len(fuera)}")

with open(sys.argv[2], 'w') as f:
    f.write(' '.join(sorted(removed)))
print(f"Total a podar: {len(removed)}")
