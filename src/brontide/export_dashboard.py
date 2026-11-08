"""Stage 10c — the self-contained interactive dashboard.

Emits a single ``dashboard.html`` with every byte of data inlined (base64 for the point
arrays, JSON for the series), so it opens from ``file://`` with no server, no network, and no
CDN. Built only from committed artifacts.

The map is downsampled from the full 149,994-point export: a 2-D canvas scatter saturates
well before that, and halving the points halves the file a reader has to download.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from . import config, export_figs

DEFAULT_POINTS = 75_000


def _payload(n_points: int, rng_seed: int = 0) -> dict:
    pts, cl, yr, meta = export_figs._map_arrays()
    rng = np.random.default_rng(rng_seed)
    n = len(cl)
    keep = np.sort(rng.choice(n, size=min(n_points, n), replace=False))

    # float64 up front: the int16 extent spans ~61.5k, which overflows int16 on subtraction
    p = pts[keep].astype(np.float64)
    x0, x1 = p[:, 0].min(), p[:, 0].max()
    y0, y1 = p[:, 1].min(), p[:, 1].max()
    # requantize into uint16 over the actual extent: same picture, no wasted range
    qx = ((p[:, 0] - x0) / (x1 - x0) * 65535).astype(np.uint16)
    qy = ((p[:, 1] - y0) / (y1 - y0) * 65535).astype(np.uint16)
    xy = np.empty(len(keep) * 2, np.uint16)
    xy[0::2], xy[1::2] = qx, qy

    figs = json.loads((config.ARTIFACTS_DIR / "region_figs.json").read_text(encoding="utf-8"))
    colors = {c["id"]: c["color"] for c in meta["clusters"]}
    for r in figs["regions"]:
        r["color"] = colors.get(r["id"], "#888888")

    return {
        "xy": base64.b64encode(xy.tobytes()).decode(),
        "cluster": base64.b64encode(cl[keep].tobytes()).decode(),
        "year": base64.b64encode(yr[keep].tobytes()).decode(),
        "n": int(len(keep)),
        "year0": meta["year0"], "year1": meta["year1"],
        "figs": figs,
    }


_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>brontide — research frontier dashboard</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--line:#232a34;--ink:#e6edf3;--ink2:#9aa4b2;--ink3:#6b7280;
--blue:#3987e5;--orange:#d95926;--aqua:#199e70;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:20px 24px 14px;border-bottom:1px solid var(--line)}
h1{margin:0 0 4px;font-size:19px;letter-spacing:-.01em}
.sub{color:var(--ink2);font-size:13px;max-width:80ch}
.chips{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
.chip{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:7px 11px}
.chip b{display:block;font-size:16px;font-variant-numeric:tabular-nums}
.chip span{color:var(--ink3);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
main{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:18px;padding:18px 24px 32px}
@media(max-width:940px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.card h2{margin:0 0 10px;font-size:13px;color:var(--ink2);font-weight:600;
text-transform:uppercase;letter-spacing:.06em}
#mapwrap{position:relative}
canvas{width:100%;display:block;border-radius:8px;background:#0a0e14;cursor:crosshair}
.controls{display:flex;align-items:center;gap:12px;margin-top:12px}
button{background:#21262d;color:var(--ink);border:1px solid var(--line);border-radius:6px;
padding:6px 14px;cursor:pointer;font:inherit}
button:hover{background:#2d333b}
input[type=range]{flex:1;accent-color:var(--blue)}
#yr{font-variant-numeric:tabular-nums;min-width:4ch;font-weight:600}
#tip{position:absolute;pointer-events:none;background:#0a0e14ee;border:1px solid var(--line);
border-radius:6px;padding:7px 10px;font-size:12px;opacity:0;transition:opacity .1s;max-width:240px}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0 0;font-size:12px;color:var(--ink2)}
.legend i{display:inline-block;width:11px;height:3px;border-radius:2px;margin-right:5px;
vertical-align:middle}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{text-align:left;color:var(--ink3);font-weight:600;padding:5px 6px;border-bottom:1px solid var(--line);
font-size:11px;text-transform:uppercase;letter-spacing:.04em}
td{padding:5px 6px;border-bottom:1px solid #1b212a}
tbody tr{cursor:pointer}
tbody tr:hover{background:#1c2230}
tbody tr.sel{background:#1f2937}
.g{font-variant-numeric:tabular-nums;text-align:right}
.sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:7px}
#list{max-height:340px;overflow-y:auto}
.note{color:var(--ink3);font-size:11.5px;margin-top:10px;line-height:1.5}
#terms{color:var(--ink3);font-size:12px;margin:2px 0 10px}
</style></head><body>
<header>
  <h1>The arXiv research frontier — <span style="color:var(--ink2)">emerging vs saturating regions</span></h1>
  <div class="sub">2.7M papers embedded with SPECTER2, cut into 1,024 microregions and 64 Ward
  super-regions, with 24-month share forecasts. The pre-registered criterion returned
  <b>0/3</b> — a published null. Every number here comes from committed artifacts.</div>
  <div class="chips">
    <div class="chip"><span>papers</span><b>2,705,992</b></div>
    <div class="chip"><span>months</span><b>234</b></div>
    <div class="chip"><span>microregions</span><b>1,024</b></div>
    <div class="chip"><span>super-regions</span><b>64</b></div>
    <div class="chip"><span>criterion</span><b>0 / 3</b></div>
    <div class="chip"><span>chance p@50</span><b>0.101</b></div>
  </div>
</header>
<main>
  <section class="card" id="mapwrap">
    <h2>Frontier map — drag the year</h2>
    <canvas id="map" width="1180" height="820"></canvas>
    <div id="tip"></div>
    <div class="controls">
      <button id="play">▶ Play</button>
      <input type="range" id="slider" min="0" max="19" value="19">
      <span id="yr">2026</span>
    </div>
    <div class="note">Each dot is a paper, positioned by SPECTER2 embedding (PCA→UMAP, display
    only) and coloured by super-region. Hover for the region under the cursor; click to inspect
    it. Points accumulate — the map shows everything published up to the selected year.</div>
  </section>
  <section class="card">
    <h2>Region detail</h2>
    <div id="rname" style="font-size:15px;font-weight:600">—</div>
    <div id="terms"></div>
    <canvas id="chart" width="820" height="430"></canvas>
    <div class="legend">
      <span><i style="background:#6f6e6a"></i>history</span>
      <span><i style="background:var(--blue)"></i>forecast (<span id="mdl">—</span>)</span>
    </div>
    <div class="note" id="rnote"></div>
  </section>
  <section class="card" style="grid-column:1/-1">
    <h2>Super-regions by forecast growth</h2>
    <div id="list"><table><thead><tr><th>region</th><th>top terms</th>
    <th class="g">ĝ</th><th class="g">emerging microregions</th></tr></thead>
    <tbody id="rows"></tbody></table></div>
    <div class="note">ĝ is the median microregion forecast-growth ratio inside each super-region.
    Produced by the CV-selected model at the production origin — which is SmoothedMomentum, the
    baseline the roster failed to beat. Read it as a ranking, not a promise.</div>
  </section>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const b64=s=>{const b=atob(s),u=new Uint8Array(b.length);
for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u;};
const XY=new Uint16Array(b64(D.xy).buffer),CL=b64(D.cluster),YR=b64(D.year);
const regions=D.figs.regions,byId={};regions.forEach(r=>byId[r.id]=r);
const cv=document.getElementById('map'),cx=cv.getContext('2d');
const W=cv.width,H=cv.height,PAD=26;
const hex=h=>[parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];
const RGB=new Uint8Array(64*3);regions.forEach(r=>{const c=hex(r.color);
RGB[r.id*3]=c[0];RGB[r.id*3+1]=c[1];RGB[r.id*3+2]=c[2];});
const px=i=>PAD+XY[i*2]/65535*(W-2*PAD), py=i=>PAD+(1-XY[i*2+1]/65535)*(H-2*PAD);
// `sel` drives the detail panel; `focus` drives map dimming and is only set by an explicit
// click, so the first paint shows the whole cloud at full strength.
let year=19,sel=null,focus=null,timer=null;
const GAIN=1.35;   // 75k points over this canvas are sparser than the full export

function draw(){
  const img=cx.createImageData(W,H),d=img.data;
  for(let i=0;i<D.n;i++){
    if(YR[i]>year)continue;
    const x=px(i)|0,y=py(i)|0;if(x<0||y<0||x>=W||y>=H)continue;
    const o=(y*W+x)*4,c=CL[i]*3,dim=(focus!==null&&CL[i]!==focus)?0.25:GAIN,
          nw=(YR[i]===year)?1.5:1;
    d[o]=Math.min(255,d[o]+RGB[c]*dim*nw);
    d[o+1]=Math.min(255,d[o+1]+RGB[c+1]*dim*nw);
    d[o+2]=Math.min(255,d[o+2]+RGB[c+2]*dim*nw);
    d[o+3]=255;
  }
  // composite over the panel ground so unpainted pixels aren't transparent black
  for(let i=3;i<d.length;i+=4){if(d[i]===0){d[i-3]=10;d[i-2]=14;d[i-1]=20;d[i]=255;}}
  cx.putImageData(img,0,0);
  document.getElementById('yr').textContent=D.year0+year;
}

function nearest(mx,my){
  let best=-1,bd=1e9;
  for(let i=0;i<D.n;i+=3){
    if(YR[i]>year)continue;
    const dx=px(i)-mx,dy=py(i)-my,dd=dx*dx+dy*dy;
    if(dd<bd){bd=dd;best=i;}
  }
  return bd<900?CL[best]:-1;
}
const tip=document.getElementById('tip');
// the hover scan touches thousands of points, so run it at most once per frame
let pending=null,rafId=0;
cv.addEventListener('mousemove',e=>{
  const r=cv.getBoundingClientRect();
  pending={mx:(e.clientX-r.left)*W/r.width,my:(e.clientY-r.top)*H/r.height,
           lx:e.clientX-r.left,ly:e.clientY-r.top,rw:r.width};
  if(rafId)return;
  rafId=requestAnimationFrame(()=>{
    rafId=0;const q=pending;if(!q)return;
    const id=nearest(q.mx,q.my);
    if(id<0){tip.style.opacity=0;return;}
    const g=byId[id];
    tip.innerHTML='<b>'+g.name+'</b><br><span style="color:#9aa4b2">'+g.terms.slice(0,5).join(', ')+'</span>';
    tip.style.left=Math.min(q.lx+14,q.rw-250)+'px';
    tip.style.top=(q.ly+14)+'px';tip.style.opacity=1;
  });
});
cv.addEventListener('mouseleave',()=>tip.style.opacity=0);
cv.addEventListener('click',e=>{
  const r=cv.getBoundingClientRect(),id=nearest((e.clientX-r.left)*W/r.width,(e.clientY-r.top)*H/r.height);
  if(id>=0){focus=id;select(id);}
});

const ch=document.getElementById('chart'),ctx=ch.getContext('2d');
function chart(r){
  const CW=ch.width,CH=ch.height,L=52,B=34,T=14,R=14;
  ctx.clearRect(0,0,CW,CH);
  if(!r)return;
  const h=r.history,f=r.forecast,all=h.concat(f);
  const lo=Math.min(...all)*0.92,hi=Math.max(...all)*1.08;
  const X=i=>L+i/(all.length-1)*(CW-L-R), Y=v=>CH-B-(v-lo)/(hi-lo)*(CH-B-T);
  ctx.strokeStyle='#232a34';ctx.lineWidth=1;ctx.fillStyle='#6b7280';
  ctx.font='11px system-ui';ctx.textAlign='right';
  for(let k=0;k<=4;k++){const v=lo+(hi-lo)*k/4,y=Y(v);
    ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(CW-R,y);ctx.stroke();
    ctx.fillText(v.toFixed(2)+'%',L-8,y+4);}
  const months=D.figs.months,tk=[0,Math.floor(h.length/2),h.length-1,all.length-1];
  tk.forEach((i,k)=>{
    ctx.textAlign=k===0?'left':(k===tk.length-1?'right':'center');
    ctx.fillText(i<h.length?months[i]:D.figs.forecast_months[i-h.length],X(i),CH-12);});
  // cutoff marker
  ctx.strokeStyle='#3d4652';ctx.setLineDash([3,3]);
  ctx.beginPath();ctx.moveTo(X(h.length-1),T);ctx.lineTo(X(h.length-1),CH-B);ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle='#6b7280';ctx.textAlign='right';
  ctx.fillText('forecast origin',X(h.length-1)-6,T+11);
  ctx.strokeStyle='#6f6e6a';ctx.lineWidth=1.4;ctx.beginPath();
  h.forEach((v,i)=>i?ctx.lineTo(X(i),Y(v)):ctx.moveTo(X(i),Y(v)));ctx.stroke();
  ctx.strokeStyle='#3987e5';ctx.lineWidth=2.2;ctx.beginPath();
  ctx.moveTo(X(h.length-1),Y(h[h.length-1]));
  f.forEach((v,i)=>ctx.lineTo(X(h.length+i),Y(v)));ctx.stroke();
}

function select(id){
  sel=id;const r=byId[id];
  document.getElementById('rname').textContent=r.name;
  document.getElementById('terms').textContent=r.terms.join(' · ');
  document.getElementById('rnote').textContent=r.sample_title?('Representative paper: “'+r.sample_title+'”'):'';
  document.getElementById('mdl').textContent=D.figs.selected_model;
  chart(r);draw();
  [...document.querySelectorAll('#rows tr')].forEach(t=>t.classList.toggle('sel',+t.dataset.id===id));
}

const rows=document.getElementById('rows');
regions.slice().sort((a,b)=>(b.g_hat??0)-(a.g_hat??0)).forEach(r=>{
  const tr=document.createElement('tr');tr.dataset.id=r.id;
  tr.innerHTML='<td><span class="sw" style="background:'+r.color+'"></span>'+r.name+'</td>'+
    '<td style="color:#6b7280">'+r.terms.slice(0,4).join(', ')+'</td>'+
    '<td class="g">'+(r.g_hat!=null?r.g_hat.toFixed(2):'—')+'</td>'+
    '<td class="g">'+r.n_emerging+'</td>';
  tr.onclick=()=>{focus=r.id;select(r.id);};rows.appendChild(tr);
});

const slider=document.getElementById('slider');
slider.oninput=()=>{year=+slider.value;draw();};
document.getElementById('play').onclick=function(){
  if(timer){clearInterval(timer);timer=null;this.textContent='▶ Play';return;}
  this.textContent='❚❚ Pause';
  if(year>=19)year=0;
  timer=setInterval(()=>{year++;slider.value=year;draw();
    if(year>=19){clearInterval(timer);timer=null;
      document.getElementById('play').textContent='▶ Play';}},420);
};
draw();
select(regions.reduce((a,b)=>((b.g_hat??0)>(a.g_hat??0)?b:a)).id);
</script></body></html>
"""


def run(out: Path = None, n_points: int = DEFAULT_POINTS) -> Path:
    """Write the self-contained dashboard. Returns its path."""
    out = Path(out or config.REPO_ROOT / "dashboard.html")
    if not (config.ARTIFACTS_DIR / "region_figs.json").exists():
        export_figs.region_figs_json()
    data = json.dumps(_payload(n_points), separators=(",", ":"))
    out.write_text(_HTML.replace("__DATA__", data), encoding="utf-8")
    return out
