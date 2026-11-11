"""Regenerate exact composite-Hilbert 4-tile K1 demo geometry; stdlib only."""
from pathlib import Path
import json,collections
p=Path(__file__).resolve().parents[1]
def hilbert(g):
 out=[]
 for d in range(g*g):
  x=y=0;t=d;s=1
  while s<g:
   rx=(t//2)&1;ry=(t^(t//2))&1
   if ry==0:
    if rx:x,y=s-1-x,s-1-y
    x,y=y,x
   x+=s*rx;y+=s*ry;t//=4;s*=2
  out.append((g-1-y)*g+x)
 return out
h=hilbert(64);order=h[1024:3072]+[4095-i for i in h[1024:3072]];inverse={v:i for i,v in enumerate(order)}
centers=[]
for b in range(64):
 ts=order[b*64:(b+1)*64];centers.append([sum(t%64 for t in ts)/64,sum(t//64 for t in ts)/64])
phases=[]
for phase in range(4):
 offset=phase*256;labels=[((i*64-offset)%4096)//1024 for i in range(64)];counts=collections.Counter()
 for y in range(64):
  for x in range(64):
   a=y*64+x
   for b in ([a+1] if x<63 else [])+([a+64] if y<63 else []):
    u,v=inverse[a]//64,inverse[b]//64
    if labels[u]!=labels[v]:counts[u,v]+=1;counts[v,u]+=1
 edges=set()
 for u in range(64):
  opts=sorted([(v,c) for (q,v),c in counts.items() if q==u],key=lambda x:(-x[1],(x[0]-offset//64)%64))[:1]
  for v,c in opts:edges.add(tuple(sorted([u,v])))
 phases.append({'labels':labels,'edges':sorted(edges),'offset':offset})
(p/'data/demo.json').write_text(json.dumps({'centers':centers,'phases':phases}))
