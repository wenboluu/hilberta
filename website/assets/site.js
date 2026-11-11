'use strict';
const $ = s => document.querySelector(s);
const ns = 'http://www.w3.org/2000/svg';
const S = (tag, attrs = {}, text) => {const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e;};
const colors=['#85898e','#476b87','#8299aa','#afb1b4','#5e646b'];
const lineStyles=['5 4','none','none','2 4','7 3 1 3'];
const tileColors=['#55758e','#7c96a9','#9bafbe','#b9c9d4'];
const names=['Full FLUX','HilbertA · 4 tiles','HilbertA · 16 tiles','CLEAR','SpargeAttn'];
let data,demo,resolution='1024',kernelResolution=0;
const precision={fid:2,kid:2,clip:4,lpips:3,psnr:2,ssim:3};
function qualityTable(selector,columns){
 const rows=data.quality[resolution],body=$(selector+' tbody');body.replaceChildren();
 const winners=columns.map(c=>{const values=rows.slice(1).map(r=>r[c.index]);const best=c.direction==='min'?Math.min(...values):Math.max(...values);return rows.map((r,i)=>i>0&&r[c.index]===best);});
 rows.forEach((r,i)=>{const tr=document.createElement('tr');tr.className=i===0?'reference':i<3?'ours':'';const th=document.createElement('th');th.scope='row';th.textContent=names[i];th.title=data.names[i];tr.append(th);
 columns.forEach((c,j)=>{const td=document.createElement('td');let value;
 if(r[c.index]===null)value='—';else if(c.index===1)value=`${(r[1]*1000).toFixed(2)} ± ${(r[2]*1000).toFixed(2)}`;else value=r[c.index].toFixed(c.digits);
 td.textContent=value;if(winners[j][i]){td.className='best';td.setAttribute('aria-label',value+'; best among sparse methods');}tr.append(td);});body.append(tr);});
}
function renderQuality(){
 qualityTable('#distribution',[{index:0,direction:'min',digits:2},{index:1,direction:'min',digits:2},{index:3,direction:'max',digits:4}]);
 qualityTable('#fidelity',[{index:4,direction:'min',digits:3},{index:5,direction:'max',digits:2},{index:6,direction:'max',digits:3}]);
 $('#resolution-note').textContent=resolution==='2048'?'2048 comparison: our LoRAs were adapted at 2048; CLEAR uses its official 1024 checkpoint; SpargeAttn is training-free.':'';
}
function renderTiming(){
 for(const res of ['1024','2048']){
  const table=$(`#time-${res} tbody`),rows=data.quality[res];table.replaceChildren();
  rows.forEach((r,i)=>{
   const tr=document.createElement('tr');tr.className=i===0?'reference':i<3?'ours':'';
   const name=document.createElement('th');name.scope='row';name.textContent=names[i];
   const seconds=document.createElement('td');seconds.textContent=r[7].toFixed(2)+' s';
   const speed=document.createElement('td');speed.textContent=i?(rows[0][7]/r[7]).toFixed(2)+'×':'—';
   if(i===2)speed.className='best';tr.append(name,seconds,speed);table.append(tr);
  });
 }
}

const tooltip=document.createElement('div');tooltip.className='chart-tooltip';tooltip.hidden=true;document.body.append(tooltip);
function hideTip(){tooltip.hidden=true;}
function showTip(text,e){tooltip.textContent=text;tooltip.hidden=false;const r=e.target.getBoundingClientRect();let x=r.left+12,y=r.top-45;x=Math.min(Math.max(8,x),innerWidth-tooltip.offsetWidth-8);y=Math.max(8,y);tooltip.style.left=x+'px';tooltip.style.top=y+'px';}
function chart(selector,key,unit){
 const root=$(selector);root.replaceChildren();const rows=data[key];const kernel=key==='kernel_ms',domain=kernel?[.1,250]:[.1,20];
 const x=[63,214,365],y=v=>270-(Math.log10(v)-Math.log10(domain[0]))/(Math.log10(domain[1])-Math.log10(domain[0]))*238;
 const ticks=kernel?[.1,1,10,100]:[.1,1,10];
 ticks.forEach(v=>{const yy=y(v);root.append(S('line',{x1:53,y1:yy,x2:409,y2:yy,stroke:'#e7e9ec','stroke-width':1}));root.append(S('text',{x:42,y:yy+4,'text-anchor':'end','font-size':11,fill:'#8b9096'},String(v)));});
 [1024,2048,4096].forEach((res,j)=>root.append(S('text',{x:x[j],y:296,'text-anchor':'middle','font-size':12,fill:'#666c73'},res+'²')));
 rows.forEach((r,i)=>{const values=r.map(v=>v===null?null:v/(kernel?1:1000));const pts=values.map((v,j)=>v===null?null:[x[j],y(v)]).filter(Boolean);
 root.append(S('polyline',{points:pts.map(p=>p.join(',')).join(' '),fill:'none',stroke:colors[i],'stroke-width':i===1||i===2?2.3:1.7,'stroke-dasharray':lineStyles[i],'stroke-linejoin':'round'}));
 values.forEach((v,j)=>{if(v===null)return;const label=`${names[i]}, ${[1024,2048,4096][j]}²: ${v.toFixed(kernel?3:3)} ${unit}${i>0?'; '+(rows[0][j]/r[j]).toFixed(2)+'× vs Full FLUX':''}`;
 const point=S('circle',{cx:x[j],cy:y(v),r:3.5,fill:colors[i],stroke:'white','stroke-width':1.2,tabindex:0,role:'img','aria-label':label,class:'chart-point'});point.append(S('title',{},label));point.addEventListener('mouseenter',e=>showTip(label,e));point.addEventListener('mouseleave',hideTip);point.addEventListener('focus',e=>showTip(label,e));point.addEventListener('blur',hideTip);root.append(point);
 if(j===2&&(i===0||i===2)){root.append(S('text',{x:x[j]+10,y:y(v)+(i===0?-7:14),fill:colors[i],'font-size':11},v.toFixed(2)));}
 });});
 if(!kernel)root.append(S('text',{x:63,y:323,fill:'#747a80','font-size':10},'CLEAR: no 4096 point (OOM)'));
}
function renderKernel(){
 const body=$('#kernel-values tbody');body.replaceChildren();data.kernel_ms.forEach((row,i)=>{const tr=document.createElement('tr');tr.className=i===0?'reference':'ours';const th=document.createElement('th');th.scope='row';th.textContent=names[i];const td=document.createElement('td');td.textContent=row[kernelResolution].toFixed(2);const speed=document.createElement('td');speed.textContent=(data.kernel_ms[0][kernelResolution]/row[kernelResolution]).toFixed(2)+'×';if(i===2){td.className='best';speed.className='best';}tr.append(th,td,speed);body.append(tr);});
}
async function initAnimation(){
 // Decode once before creating the SVG's 64 crops or starting its clock.
 const picture=new Image();picture.src='assets/patch-demo.jpg';await picture.decode();
 const root=$('#method-animation'),reduce=matchMedia('(prefers-reduced-motion: reduce)'),mobile=matchMedia('(max-width:600px)');
 let playing=!reduce.matches,inView=false,last=null,time=0,lastKey='',manualPhase=null;
 const starts=[0,4200,8200,17000],durations=[4200,4000,8800,6000],total=23000;
 const grid=i=>[42+(demo.centers[i][0]-3.5)/8*38,58+(demo.centers[i][1]-3.5)/8*38];
 const seq=i=>[42+(i%8)*38,58+Math.floor(i/8)*38];
 const defs=S('defs');root.append(defs);
 const arrow=S('marker',{id:'bridge-arrow',viewBox:'0 0 10 10',refX:7,refY:5,markerWidth:2.5,markerHeight:2.5,orient:'auto-start-reverse'});arrow.append(S('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'#956653'}));defs.append(arrow);
 const left=S('g'),right=S('g');root.append(left,right);
 const leftTitle=S('text',{x:192,y:27,'text-anchor':'middle',class:'diagram-title'});left.append(leftTitle);
 const tokenGroup=S('g');left.append(tokenGroup);const tokens=[];
 for(let i=0;i<64;i++){
  const x=(demo.centers[i][0]-3.5)/8,y=(demo.centers[i][1]-3.5)/8;
  const g=S('g',{class:'token'});g.style.transform=`translate(${grid(i)[0]}px,${grid(i)[1]}px)`;const crop=S('svg',{width:34,height:34,viewBox:`${x*80} ${y*80} 80 80`,overflow:'hidden'});
  crop.append(S('image',{href:'assets/patch-demo.jpg',width:640,height:640,class:'picture-patch'}));
  const tint=S('rect',{width:34,height:34,fill:'transparent','fill-opacity':.24});
  const border=S('rect',{width:34,height:34,fill:'none',stroke:'white','stroke-width':.6});
  g.append(crop,tint,border,S('title',{},`Block ${i}: 64 image tokens`));tokenGroup.append(g);tokens.push({g,tint,border});
 }
 const frames=S('g');left.append(frames);
 const curve=S('polyline',{class:'curve-line',points:demo.centers.map((_,i)=>grid(i).map(v=>v+17).join(',')).join(' '),stroke:'#fff','stroke-width':2});left.append(curve);
 const length=curve.getTotalLength();curve.style.strokeDasharray=length;
 const dot=S('circle',{r:5,fill:'#fff',stroke:'#202124','stroke-width':2});left.append(dot);
 const bridge=S('g');left.append(bridge);
 const seqHint=S('text',{x:192,y:382,'text-anchor':'middle',class:'diagram-label'});left.append(seqHint);
 const connector=S('path',{d:'M 388 208 H 456 M 446 198 L 456 208 L 446 218',fill:'none',stroke:'#9ba1a8','stroke-width':2});root.append(connector);
 const matrixTitle=S('text',{x:665,y:27,'text-anchor':'middle',class:'diagram-title'});right.append(matrixTitle);
 right.append(S('text',{x:665,y:57,'text-anchor':'middle',class:'diagram-label'},'Keys →'),S('text',{x:502,y:204,transform:'rotate(90 502 204)','text-anchor':'middle',class:'diagram-label'},'Queries →'));
 const matrix=S('g',{'shape-rendering':'crispEdges'});right.append(matrix);const cells=[];
 for(let q=0;q<64;q++)for(let k=0;k<64;k++){const r=S('rect',{x:537+k*4,y:72+q*4,width:4.05,height:4.05});matrix.append(r);cells.push(r);}
 const matrixBounds=S('g');right.append(matrixBounds);
 const row=S('rect',{x:537,width:256,height:4,fill:'none',stroke:'#202124','stroke-width':1.5});right.append(row);
 const callout=S('g');right.append(callout);
 const amount=S('text',{x:665,y:366,'text-anchor':'middle',class:'diagram-number'});right.append(amount);
 const captions=['Keep nearby image patches together along the Hilbert curve.','Each tile computes dense attention. Empty matrix regions are skipped.','Shift tile boundaries each layer. The marked block gets new neighbors.','A bridge adds attention in both directions between neighboring blocks.'];
 function resize(){const small=mobile.matches;root.setAttribute('viewBox',small?'0 0 400 620':'0 0 900 405');left.setAttribute('transform',small?'translate(50 0) scale(.75)':'');right.setAttribute('transform',small?'translate(-300 300) scale(.75)':'');connector.style.display=small?'none':'';lastKey='';}
 mobile.addEventListener('change',resize);resize();
 function updatePlay(){const b=$('#play');b.setAttribute('aria-label',playing?'Pause animation':'Play animation');b.setAttribute('aria-pressed',String(playing));b.firstElementChild.textContent=playing?'Ⅱ':'▶';root.classList.toggle('paused',!playing);}
 function draw(){
  const stage=starts.reduce((s,v,i)=>time>=v?i:s,0),elapsed=time-starts[stage];
  const phase=manualPhase===null?(stage===2?Math.min(3,Math.floor(elapsed/2200)):0):manualPhase;
  const plan=demo.phases[phase],labels=plan.labels;
  // Use an actual spatially adjacent bridge; keep that query fixed through sliding.
  const edge=plan.edges.find(([a,b])=>{const aa=grid(a),bb=grid(b);return Math.abs(aa[0]-bb[0])+Math.abs(aa[1]-bb[1])===38;})||plan.edges[0];
  const query=stage===3?edge[0]:15,target=edge[1],key=`${stage}-${phase}`;
  if(key!==lastKey){
   const inSequence=stage===1||stage===2;
   tokens.forEach(({g,tint,border},i)=>{
    const pos=inSequence?seq(i):grid(i);g.style.transform=`translate(${pos[0]}px,${pos[1]}px)`;
    tint.setAttribute('fill',stage===0?'transparent':tileColors[labels[i]]);
    border.setAttribute('stroke',stage===0?'white':tileColors[labels[i]]);border.setAttribute('stroke-width',stage===0?.6:2);
    g.style.opacity=stage===3&&labels[i]!==labels[query]&&i!==target?'.42':'1';
   });
   frames.replaceChildren();bridge.replaceChildren();callout.replaceChildren();
   if(inSequence){
    // These are pieces of the same cyclic sequence. Outlines follow exact phase membership.
    for(let r=0;r<8;r++){let start=0;while(start<8){let end=start+1;const t=labels[r*8+start];while(end<8&&labels[r*8+end]===t)end++;
     frames.append(S('rect',{x:40+start*38,y:56+r*38,width:(end-start)*38,height:38,rx:2,fill:'none',stroke:tileColors[t],'stroke-width':3}));start=end;}}
    if(stage===2){const pos=seq(query);frames.append(S('rect',{x:pos[0]-1,y:pos[1]-1,width:36,height:36,fill:'none',stroke:'#17191c','stroke-width':3}));frames.append(S('circle',{cx:pos[0]+17,cy:pos[1]+17,r:6,fill:'white',stroke:'#17191c','stroke-width':2}));}
   }
   const extras=new Set(plan.edges.flatMap(([a,b])=>[a*64+b,b*64+a]));let count=0;
   cells.forEach((cell,index)=>{const q=Math.floor(index/64),k=index%64,same=labels[q]===labels[k],extra=stage===3&&extras.has(index),on=stage===0||same||extra;if(on)count++;
    cell.setAttribute('fill',stage===0?'#c6cbd0':extra?'#956653':same?tileColors[labels[q]]:'#f1f2f4');cell.setAttribute('opacity','1');});
   matrixBounds.replaceChildren();if(stage>0){const runs=[];let start=0;while(start<64){let end=start+1;while(end<64&&labels[end]===labels[start])end++;runs.push([start,end,labels[start]]);start=end;}for(const [q0,q1,t] of runs)for(const [k0,k1,u] of runs)if(t===u)matrixBounds.append(S('rect',{x:537+k0*4,y:72+q0*4,width:(k1-k0)*4,height:(q1-q0)*4,fill:'none',stroke:'#6c8293','stroke-width':.65,opacity:.6}));}
   row.setAttribute('y',72+query*4);row.style.display=stage===2?'':'none';
   if(stage===3){
    const a=grid(query).map(v=>v+17),b=grid(target).map(v=>v+17);
    for(const [i,label] of [[query,'A'],[target,'B']]){const pos=grid(i);bridge.append(S('rect',{x:pos[0]-2,y:pos[1]-2,width:38,height:38,fill:'none',stroke:'#956653','stroke-width':3}));bridge.append(S('circle',{cx:pos[0]+17,cy:pos[1]+17,r:10,fill:'#fff'}),S('text',{x:pos[0]+17,y:pos[1]+21,'text-anchor':'middle','font-size':12,'font-weight':700,fill:'#815441'},label));}
    const distance=Math.hypot(b[0]-a[0],b[1]-a[1]),dx=(b[0]-a[0])/distance,dy=(b[1]-a[1])/distance;
    bridge.insertBefore(S('line',{x1:a[0]+dx*12,y1:a[1]+dy*12,x2:b[0]-dx*12,y2:b[1]-dy*12,stroke:'#956653','stroke-width':3,'marker-start':'url(#bridge-arrow)','marker-end':'url(#bridge-arrow)'}),bridge.firstChild);
    for(const [q,k,label] of [[query,target,'A → B'],[target,query,'B → A']]){const x=537+k*4,y=72+q*4;callout.append(S('rect',{x:x-2,y:y-2,width:8,height:8,fill:'#956653',stroke:'#fff','stroke-width':1}));const tx=q<k?815:514;callout.append(S('line',{x1:x+2,y1:y+2,x2:tx,y2:y+2,stroke:'#956653','stroke-width':1}),S('text',{x:tx,y:y-5,'text-anchor':q<k?'start':'end',fill:'#815441','font-size':12,'font-weight':600},label));}
   }
   leftTitle.textContent=stage===0?'Image → Hilbert order':inSequence?'One sequence · four tiles':'One bridge, two directions';
   matrixTitle.textContent=stage===0?'Full image attention':stage===3?'Image attention · all bridges':'Computed image attention';
   seqHint.textContent=inSequence?'Sequence order → (continues on next row)':'';
   amount.textContent=stage===0?'100% of image pairs':`${(count/4096*100).toFixed(stage===3?1:0)}% of image pairs`;
   $('#animation-caption').textContent=captions[stage];$('.phase-controls').hidden=false;$('.phase-controls').style.visibility=stage<2?'hidden':'visible';$('.phase-controls').setAttribute('aria-hidden',String(stage<2));
   document.querySelectorAll('[data-stage]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.stage)===stage)));
   document.querySelectorAll('[data-phase]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.phase)===phase)));
   lastKey=key;
  }
  const settled=!playing||elapsed>=900;
  const explaining=settled||stage===0||stage===2;right.style.opacity=explaining?'1':'0';$('#animation-caption').style.opacity=explaining?'1':'0';
  frames.style.opacity=settled?'1':'0';bridge.style.opacity=settled?'1':'0';callout.style.opacity=settled?'1':'0';
  curve.style.opacity=stage===0&&settled?'1':'0';dot.style.opacity=stage===0&&settled?'1':'0';
  if(stage===0){const t=Math.max(0,Math.min(1,(elapsed-900)/2700));curve.style.strokeDashoffset=length*(1-t);const p=curve.getPointAtLength(length*t);dot.setAttribute('cx',p.x);dot.setAttribute('cy',p.y);}
  document.querySelectorAll('[data-stage] i').forEach((el,i)=>el.style.transform=`scaleX(${i<stage?1:i>stage?0:Math.min(1,elapsed/durations[i])})`);
  root.dataset.stage=stage;root.dataset.phase=phase;root.dataset.playing=playing;root.dataset.entries=stage===0?4096:1024+(stage===3?2*plan.edges.length:0);
 }
 function frame(now){if(last!==null&&playing&&inView&&!document.hidden)time=(time+Math.min(now-last,100))%total;last=now;draw();requestAnimationFrame(frame);}
 $('#play').addEventListener('click',()=>{playing=!playing;if(playing)manualPhase=null;updatePlay();draw();});
 document.querySelectorAll('[data-stage]').forEach(b=>b.addEventListener('click',()=>{playing=false;manualPhase=null;time=starts[Number(b.dataset.stage)]+1600;updatePlay();draw();}));
 document.querySelectorAll('[data-phase]').forEach(b=>b.addEventListener('click',()=>{playing=false;manualPhase=Number(b.dataset.phase);updatePlay();draw();}));
 reduce.addEventListener('change',()=>{if(reduce.matches){playing=false;updatePlay();}});
 new IntersectionObserver(entries=>{inView=entries[0].intersectionRatio>=.9;},{threshold:.9}).observe($('#method'));
 if(reduce.matches)time=4000;updatePlay();draw();
 await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
 root.dataset.ready='true';requestAnimationFrame(frame);
}
Promise.all(['results','demo'].map(name=>fetch(`data/${name}.json`).then(r=>{if(!r.ok)throw Error(name);return r.json();}))).then(([results,geometry])=>{
 data=results;demo=geometry;renderQuality();chart('#kernel-chart','kernel_ms','ms');chart('#denoiser-chart','denoiser_ms','s');renderKernel();renderTiming();initAnimation().catch(err=>{console.error(err);$('#animation-caption').textContent='The illustration could not load. Please reload the page.';});
 ['kernel','denoiser'].forEach(kind=>names.slice(0,kind==='kernel'?3:5).forEach((name,i)=>{const span=document.createElement('span'),line=document.createElement('i');line.style.borderColor=colors[i];line.style.borderTopStyle=i===0||i>=3?'dashed':'solid';span.append(line,document.createTextNode(name));$('#'+kind+'-legend').append(span);}));
 document.querySelectorAll('[data-resolution]').forEach(b=>b.addEventListener('click',()=>{resolution=b.dataset.resolution;document.querySelectorAll('[data-resolution]').forEach(x=>x.setAttribute('aria-pressed',String(x.dataset.resolution===resolution)));kernelResolution=resolution==='1024'?0:1;document.querySelectorAll('[data-kernel-res]').forEach(x=>x.setAttribute('aria-pressed',String(Number(x.dataset.kernelRes)===kernelResolution)));renderQuality();renderKernel();}));
 document.querySelectorAll('[data-kernel-res]').forEach(b=>b.addEventListener('click',()=>{kernelResolution=Number(b.dataset.kernelRes);document.querySelectorAll('[data-kernel-res]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));renderKernel();}));
}).catch(err=>{console.error(err);$('#load-error').hidden=false;});
