import os, sys, collections
sys.path.append('/usr/share/sumo/tools'); import sumolib
net = sumolib.net.readNet(sys.argv[1], withInternal=False)
E = net.getEdges(); km = sum(e.getLength() for e in E)/1000
car = [e for e in E if e.allows('passenger')]
tram = [e for e in E if e.allows('tram')]
bus = [e for e in E if e.allows('bus')]
bici = [e for e in E if e.allows('bicycle') and not e.allows('passenger')]
oneway = sum(1 for e in car if not any(o.getToNode()==e.getFromNode() and o.getFromNode()==e.getToNode() for o in e.getToNode().getOutgoing()))
# componente fuertemente conexa para autos (Tarjan iterativo simple via Kosaraju)
def scc(edges):
    S=set(edges); succ={e:[n for n in e.getOutgoing() if n in S] for e in edges}
    pred=collections.defaultdict(list)
    for e,ns in succ.items():
        for n in ns: pred[n].append(e)
    seen=set(); order=[]
    for s in edges:
        if s in seen: continue
        stack=[(s,iter(succ[s]))]; seen.add(s)
        while stack:
            v,it=stack[-1]
            for w in it:
                if w not in seen: seen.add(w); stack.append((w,iter(succ[w]))); break
            else: order.append(v); stack.pop()
    comp={}; best=0
    for s in reversed(order):
        if s in comp: continue
        c=[s]; comp[s]=s; i=0
        while i<len(c):
            for w in pred[c[i]]:
                if w not in comp: comp[w]=s; c.append(w)
            i+=1
        best=max(best,len(c))
    return best
xmin,ymin,xmax,ymax=net.getBoundary()
ob=net.getLocationOrigBoundary() if hasattr(net,"getLocationOrigBoundary") else None
print(f"extension: {(xmax-xmin)/1000:.1f} x {(ymax-ymin)/1000:.1f} km")
print(f"aristas {len(E)}  km {km:.0f}  nodos {len(net.getNodes())}  semaforos {len(net.getTrafficLights())}")
print(f"autos {len(car)} (sentido unico {oneway})  bus {len(bus)}  solo-bici {len(bici)}  tranvia {len(tram)} ({sum(e.getLength() for e in tram)/1000:.1f} km)")
print(f"autos en la mayor componente fuertemente conexa: {scc(car)}/{len(car)}")
names={e.getName() for e in E}
for n in ['Avenida de las Américas','Avenida Remigio Crespo Toral','Avenida Fray Vicente Solano','Avenida España','Avenida Huayna Cápac','Avenida Ordóñez Lasso','Avenida Doce de Abril','Avenida Solano']:
    print(' ',n,'OK' if n in names else '--')
print('  circunvalacion:', sorted(n for n in names if 'ircunval' in n)[:5])
