"""Exporta la red SUMO a JSON (se corre dentro del contenedor, usa sumolib).
Uso: python3 escenario/exportar_red.py red/cuenca_urbana.net.xml escenario/red_info.json"""
import sys, json
sys.path.append('/usr/share/sumo/tools'); import sumolib
net = sumolib.net.readNet(sys.argv[1])
out = {'boundary': list(net.getBoundary()), 'offset': list(net.getLocationOffset()), 'edges': [], 'tls': []}
for e in net.getEdges():
    if not e.allows('passenger'): continue
    sh = e.getShape(); n = len(sh)
    # punto medio a lo largo de la forma
    L = sum(((sh[i+1][0]-sh[i][0])**2+(sh[i+1][1]-sh[i][1])**2)**.5 for i in range(n-1)); acc = 0; mid = sh[0]
    for i in range(n-1):
        d = ((sh[i+1][0]-sh[i][0])**2+(sh[i+1][1]-sh[i][1])**2)**.5
        if acc + d >= L/2:
            f = (L/2-acc)/d if d else 0; mid = (sh[i][0]+f*(sh[i+1][0]-sh[i][0]), sh[i][1]+f*(sh[i+1][1]-sh[i][1])); break
        acc += d
    out['edges'].append({'id': e.getID(), 'name': e.getName(), 'type': e.getType(), 'len': round(e.getLength(), 2),
        'lanes': e.getLaneNumber(), 'speed': round(e.getSpeed(), 2), 'prio': e.getPriority(),
        'fringe': bool(e.is_fringe()), 'mid': [round(mid[0], 1), round(mid[1], 1)],
        'shape': [[round(x, 1), round(y, 1)] for x, y in sh], 'from': e.getFromNode().getID(), 'to': e.getToNode().getID()})
for t in net.getTrafficLights():
    xs = [c[0].getEdge().getToNode().getCoord() for c in t.getConnections()]
    if xs: out['tls'].append([round(sum(x for x, _ in xs)/len(xs), 1), round(sum(y for _, y in xs)/len(xs), 1)])
json.dump(out, open(sys.argv[2], 'w'))
print('aristas', len(out['edges']), 'semaforos', len(out['tls']))
