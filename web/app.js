/* brontide atlas. Static, first-party WebGL; no network services or runtime dependencies. */
const $ = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const sum = (a) => a.reduce((s, v) => s + Number(v || 0), 0);
const mean = (a) => a.length ? sum(a) / a.length : 0;
const integer = new Intl.NumberFormat('en-US');
const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 });
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const base = new URL('../artifacts/explorer/', import.meta.url);
const state = { mode: innerWidth < 620 ? 'map' : 'terrain', from: 0, to: 19, compare: false, selected: null, micro: null, playing: false, tour: -1 };
let data, manifest, regions, byId, pointYears, pointClusters, positions, colors, renderer, microData, microPromise;
let year0 = 2007, yearMax = 19, timer, toastTimer, urlTimer, chartSeries = null;
const camera = { yaw: -.32, pitch: .46, zoom: 1, panX: 0, panY: 0 };
const targetCamera = { ...camera };

function notify(message) {
  $('toast').textContent = message; $('toast').classList.add('visible');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').classList.remove('visible'), 3500);
}
async function getFile(name, binary = false) {
  const response = await fetch(new URL(name, base));
  if (!response.ok) throw new Error(`Could not load ${name} (${response.status}).`);
  return binary ? response.arrayBuffer() : response.json();
}
function hsl(h, s, l) {
  const f = (n) => { const k = (n + h * 12) % 12; return l - s * Math.min(l, 1 - l) * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
  return [f(0), f(8), f(4)];
}
function regionColor(id) { return hsl((.43 + id * .61803398875) % 1, .57, .67); }
function cssColor(id) { return `rgb(${regionColor(id).map((v) => Math.round(v * 255)).join(',')})`; }
function readURL() {
  const previousParent = state.selected;
  const p = new URLSearchParams(location.hash.slice(1));
  if (['space', 'terrain', 'map'].includes(p.get('view'))) state.mode = p.get('view');
  const yearParam = (name, fallback) => { const n = Number(p.get(name)); return p.has(name) && Number.isFinite(n) ? Math.round(n) : fallback; };
  state.from = clamp(yearParam('from',year0) - year0, 0, yearMax);
  state.to = clamp(yearParam('to',year0+yearMax) - year0, state.from, yearMax);
  state.compare = p.get('compare') === '1';
  const id = p.has('region') ? Number(p.get('region')) : null;
  state.selected = byId.has(id) ? id : null;
  state.micro = p.has('micro') && state.selected !== null ? Number(p.get('micro')) : null;
  if (previousParent !== state.selected) { $('subarea-content').hidden = true; $('subarea-toggle').setAttribute('aria-expanded','false'); $('microregion-select').replaceChildren(); }
  if (renderer?.fallback) state.mode = 'map';
  resetCamera(false);
  for (const [param, key, lo, hi] of [['yaw','yaw',-20,20],['pitch','pitch',.08,1.57],['zoom','zoom',.45,5],['px','panX',-4,4],['py','panY',-4,4]]) {
    if (p.has(param) && Number.isFinite(Number(p.get(param)))) targetCamera[key] = clamp(Number(p.get(param)), lo, hi);
  }
  Object.assign(camera, targetCamera);
}
function writeURL(immediate = false) {
  clearTimeout(urlTimer);
  const save = () => {
    const p = new URLSearchParams({ view: state.mode, from: year0 + state.from, to: year0 + state.to });
    if (state.selected !== null) p.set('region', state.selected);
    if (state.micro !== null) p.set('micro', state.micro);
    if (state.compare) p.set('compare', '1');
    for (const [param, key] of [['yaw','yaw'],['pitch','pitch'],['zoom','zoom'],['px','panX'],['py','panY']]) p.set(param, targetCamera[key].toFixed(3));
    history.replaceState(null, '', `${location.pathname}${location.search}#${p}`);
  };
  if (immediate) save(); else urlTimer = setTimeout(save, 220);
}
function resetCamera(redraw = true) {
  Object.assign(targetCamera, { yaw: state.mode === 'map' ? 0 : -.32, pitch: state.mode === 'map' ? Math.PI / 2 : state.mode === 'terrain' ? .78 : .46, zoom: 1, panX: 0, panY: 0 });
  if (redraw) { renderer?.request(); writeURL(); }
}
function monthIndices() {
  const first = year0 + state.from, last = year0 + state.to;
  return data.months.map((m, i) => Number(m.slice(0, 4)) >= first && Number(m.slice(0, 4)) <= last ? i : -1).filter((i) => i >= 0);
}
function rawShareFor(r, i) { return r.raw_share?.[i] ?? (data.monthly_totals[i] ? (r.raw_counts[i] / data.monthly_totals[i]) * 100 : 0); }
function yearShare(r, year) { return mean(data.months.flatMap((m, i) => Number(m.slice(0, 4)) === year ? [rawShareFor(r, i)] : [])); }
function observedChange(r) { return yearShare(r, year0 + state.to) - yearShare(r, year0 + state.from); }
function setView(mode) {
  if (renderer?.fallback && mode !== 'map') { notify('This device is using the accessible 2D renderer. All topic and time controls remain available.'); return; }
  state.mode = mode; resetCamera(); updateControls(); updateDetails(); renderer?.request(); writeURL();
}
function selectRegion(id, { announce = true } = {}) {
  state.selected = id !== null && byId.has(Number(id)) ? Number(id) : null; state.micro = null;
  $('subarea-content').hidden = true; $('subarea-toggle').setAttribute('aria-expanded', 'false');
  updateTopicList(); updateDetails(); renderer.request(); writeURL();
  if (announce && state.selected !== null && innerWidth < 1000) notify(`${byId.get(state.selected).name} selected. Details are below the map.`);
}
function updateControls() {
  document.querySelectorAll('[data-view]').forEach((b) => { const active = b.dataset.view === state.mode; b.classList.toggle('active', active); b.setAttribute('aria-pressed', String(active)); });
  $('year-from').value = state.from; $('year-to').value = state.to;
  $('year-from-label').textContent = year0 + state.from; $('year-to-label').textContent = year0 + state.to;
  $('year-from').setAttribute('aria-valuetext', String(year0 + state.from)); $('year-to').setAttribute('aria-valuetext', String(year0 + state.to));
  $('window-label').textContent = `${year0 + state.from} — ${year0 + state.to}`;
  $('compare-toggle').checked = state.compare;
  $('view-caption').textContent = state.mode === 'space' ? 'VERTICAL AXIS · PUBLICATION YEAR' : state.mode === 'terrain' ? 'HEIGHT · RELATIVE SAMPLE DENSITY' : 'FIXED SEMANTIC PROJECTION · 2D';
  $('gesture-caption').textContent = state.mode === 'map' ? 'Drag to pan · scroll to zoom' : 'Drag to orbit · scroll to zoom';
  const visible = pointYears.reduce((n, y) => n + (y <= state.to && (state.compare || y >= state.from) ? 1 : 0), 0);
  $('visible-points').textContent = `${compact.format(visible)} sampled dots`;
  const totalByYear = Array.from({ length: yearMax + 1 }, () => 0);
  data.months.forEach((m, i) => totalByYear[Number(m.slice(0, 4)) - year0] += data.monthly_totals[i]);
  const max = Math.max(...totalByYear);
  $('timeline').innerHTML = totalByYear.map((v, y) => {
    const x = y * (760 / totalByYear.length), w = 760 / totalByYear.length - 5, h = v / max * 30;
    const active = y >= state.from && y <= state.to, old = state.compare && y < state.from;
    return `<rect x="${x.toFixed(2)}" y="${34-h}" width="${w}" height="${h}" rx="1" fill="${active ? '#8dd9b8' : old ? '#53696e' : '#24333e'}" opacity="${active ? '.65' : '.6'}"/>`;
  }).join('');
}
function updateTopicList() {
  const q = $('topic-search').value.trim().toLowerCase();
  const list = regions.filter((r) => `${r.name} ${r.terms.join(' ')}`.toLowerCase().includes(q));
  const sort = $('topic-sort').value;
  list.sort(sort === 'name' ? (a,b) => a.name.localeCompare(b.name) : sort === 'change' ? (a,b) => observedChange(b) - observedChange(a) : (a,b) => b.n_papers - a.n_papers);
  $('topic-list-caption').textContent = q ? `${list.length} MATCH${list.length === 1 ? '' : 'ES'}` : 'ALL TERRITORIES';
  $('topic-list').innerHTML = list.length ? list.map((r) => `<div role="listitem"><button class="topic-row${state.selected === r.id ? ' selected' : ''}" data-region="${r.id}" aria-pressed="${state.selected === r.id}"><span class="topic-swatch" style="background:${cssColor(r.id)}"></span><span class="topic-name">${esc(r.name)}</span><span class="topic-size">${sort === 'change' ? `${observedChange(r) >= 0 ? '+' : ''}${observedChange(r).toFixed(1)}pp` : compact.format(r.n_papers)}</span></button></div>`).join('') : '<p class="empty-topics">No matching territory. Try a broader keyword.</p>';
}
function updateDetails() {
  const region = byId.get(state.selected), micro = state.micro !== null ? microData?.regions.find((r) => r.id === state.micro && r.superregion_id === state.selected) : null;
  const r = micro || region, indices = monthIndices();
  const count = sum(indices.map((i) => r ? r.raw_counts[i] : data.monthly_totals[i]));
  const denominator = sum(indices.map((i) => data.monthly_totals[i]));
  $('clear-selection').hidden = !region;
  $('detail-kicker').textContent = micro ? `MICROREGION ${micro.id} · STATISTICS` : region ? `TERRITORY ${String(region.id + 1).padStart(2,'0')} / ${regions.length}` : 'THE BIG PICTURE';
  $('detail-name').textContent = micro ? `${region.name} / ${micro.name}` : region?.name || 'A world of research';
  $('detail-description').textContent = micro ? 'A statistical subarea of this territory. The map continues to highlight its parent; individual point assignments are unavailable.' : region ? `${integer.format(region.n_papers)} corpus papers in ${region.microregion_count} microregions. Explore the observed history and its experimental projection.` : state.mode === 'space' ? 'Every point is a sampled paper. Related work forms a territory; height reveals when it appeared.' : state.mode === 'terrain' ? 'Peaks reveal concentrations of the display sample in your time window. Choose a territory to inspect its monthly history.' : 'Related papers share a neighborhood. Choose a territory to follow its history through the corpus.';
  $('detail-terms').innerHTML = region && !micro ? region.terms.slice(0,6).map((t) => `<span>${esc(t)}</span>`).join('') : '';
  $('stat-one').textContent = compact.format(count); $('stat-one').title = integer.format(count);
  $('stat-two-label').textContent = r ? 'WINDOW SHARE' : 'OBSERVED PERIOD';
  $('stat-two').textContent = r ? `${denominator ? (count/denominator*100).toFixed(2) : '0.00'}%` : `${year0 + state.from}–${String(year0 + state.to).slice(2)}`;
  $('chart-title').textContent = r ? 'Smoothed monthly share' : 'Monthly publications';
  $('chart-unit').textContent = r ? '% of arXiv' : 'papers';
  $('forecast-legend').hidden = !r;
  $('chart-date-label').textContent = r ? `${year0} — ${data.forecast_months.at(-1).slice(0,4)}` : `${year0} — ${year0 + yearMax}`;
  drawChart(r);
  $('comparison-note').hidden = !(state.compare && r);
  if (r && state.compare) {
    const a = yearShare(r, year0 + state.from), b = yearShare(r, year0 + state.to), delta = b-a;
    $('comparison-note').innerHTML = `<strong>${delta >= 0 ? '+' : ''}${delta.toFixed(2)} percentage points</strong><br>Mean monthly observed share: ${a.toFixed(2)}% in ${year0+state.from} → ${b.toFixed(2)}% in ${year0+state.to}. Endpoint years use available months; ${year0+yearMax} is partial.`;
  }
  const representative = region?.representatives?.[0];
  $('representative').hidden = !representative || !!micro;
  if (representative) {
    const p = $('representative-title'); p.replaceChildren();
    const a = document.createElement('a'); a.textContent = representative.title;
    // Exports supply title-search links, never a guessed exact paper identifier.
    if (representative.url?.startsWith('https://arxiv.org/')) { a.href = representative.url; a.target = '_blank'; a.rel = 'noopener'; a.title = 'Search arXiv for this title'; p.append(a); }
    else p.textContent = representative.title;
  }
  $('subarea-section').hidden = !region; $('subarea-count').textContent = region ? `(${region.microregion_count})` : '';
  $('detail-provenance').textContent = `Frozen ${manifest.frozen_result.corpus_end} · ${r ? data.selected_model + ' forecast' : 'aggregate observations'}`;
}
function drawChart(r) {
  const historyValues = r ? r.history : data.monthly_totals, forecastValues = r ? r.forecast : [];
  const values = [...historyValues, ...forecastValues], width = 500, height = 160, L = 40, R = 8, T = 9, B = 26;
  const max = Math.max(...values) * 1.08 || 1, X = (i) => L + i / (values.length-1) * (width-L-R), Y = (v) => height-B-v/max*(height-T-B);
  const line = (a, start = 0) => a.map((v,i) => `${i ? 'L' : 'M'}${X(i+start).toFixed(2)},${Y(v).toFixed(2)}`).join(' ');
  const indices = monthIndices(), first = indices[0] || 0, last = indices.at(-1) || 0;
  let html = '<defs><linearGradient id="chart-gradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#a7edcf" stop-opacity=".14"/><stop offset="100%" stop-color="#a7edcf" stop-opacity="0"/></linearGradient></defs>';
  html += `<rect class="chart-window" x="${X(first)}" y="${T}" width="${Math.max(1,X(last)-X(first))}" height="${height-T-B}"/>`;
  for(let k=0;k<3;k++){const v=max*k/2,y=Y(v);html+=`<line class="chart-grid" x1="${L}" x2="${width-R}" y1="${y}" y2="${y}"/><text class="chart-axis" x="${L-7}" y="${y+4}" text-anchor="end">${r ? v.toFixed(v<1?2:1) : compact.format(v)}</text>`;}
  html += `<path class="chart-area" d="${line(historyValues)} L${X(historyValues.length-1)},${height-B} L${L},${height-B}Z"/><path class="chart-history" d="${line(historyValues)}"/>`;
  if(forecastValues.length){html+=`<path class="chart-forecast" d="${line([historyValues.at(-1),...forecastValues],historyValues.length-1)}"/><line class="chart-marker" x1="${X(historyValues.length-1)}" x2="${X(historyValues.length-1)}" y1="${T}" y2="${height-B}"/>`;}
  for(const i of [first,last]) html += `<line class="chart-marker" x1="${X(i)}" x2="${X(i)}" y1="${T}" y2="${height-B}"/>`;
  html += `<text class="chart-axis" x="${L}" y="${height-4}">${year0}</text>`;
  if(r){const origin=X(historyValues.length-1);html+=`<text class="chart-origin" x="${origin-6}" y="${height-B-7}" transform="rotate(-90 ${origin-6} ${height-B-7})">JUN 2026 · FORECAST ORIGIN</text><text class="chart-axis" x="${width-R}" y="${height-4}" text-anchor="end">${data.forecast_months.at(-1)}</text>`;}
  else html+=`<text class="chart-axis" x="${width-R}" y="${height-4}" text-anchor="end">${manifest.frozen_result.corpus_end}</text>`;
  html += '<g id="chart-hover" style="display:none"><line stroke="#d7f3e4" stroke-width="1" y1="8" y2="134"/><circle r="3" fill="#d7f3e4"/><rect y="0" height="23" rx="3" fill="#19382e"/><text y="16" fill="#d1e9de" font-size="12" font-family="monospace"/></g>';
  $('history-chart').innerHTML = html;
  $('history-chart').setAttribute('aria-label', r ? `${r.name}: smoothed monthly share, observed ${data.months[0]} through ${data.months.at(-1)}; experimental ${data.selected_model} forecast through ${data.forecast_months.at(-1)}.` : `Monthly corpus publications from ${data.months[0]} to ${data.months.at(-1)}.`);
  chartSeries = { values, X, Y, months:[...data.months,...(r?data.forecast_months:[])], observed:historyValues.length, unit:r?'%':' papers', L, width:width-L-R };
}
async function openMicroregions(forceOpen = false) {
  if (state.selected === null) return;
  const open = forceOpen === true || $('subarea-content').hidden;
  $('subarea-content').hidden = !open; $('subarea-toggle').setAttribute('aria-expanded',String(open));
  if (!open) {state.micro=null;updateDetails();writeURL();return;}
  $('microregion-description').textContent = 'Loading the published microregion series…';
  try {
    microPromise ||= getFile(manifest.files.microregions);
    microData = await microPromise;
    if (state.selected === null) return;
    const list = microData.regions.filter((r) => r.superregion_id === state.selected).sort((a,b)=>b.n_papers-a.n_papers);
    $('microregion-select').innerHTML = `<option value="">Whole territory</option>${list.map((r)=>`<option value="${r.id}">${esc(r.name)} · ${compact.format(r.n_papers)} papers</option>`).join('')}`;
    const validMicro = list.some((r)=>r.id===state.micro);
    if(!validMicro)state.micro=null;
    $('microregion-select').value = state.micro ?? '';
    $('microregion-description').textContent = `${list.length} k-means microregions within this Ward superregion. Names identify indices, not independently labelled research topics.`;
    updateDetails();
  } catch(error) {microPromise=null;$('microregion-description').textContent='The optional microregion bundle could not be loaded. Close and reopen to retry.';notify('Could not load microregion statistics. The atlas is still available.');}
}

// Shared camera projection is used by WebGL, labels, picking, and the 2D fallback.
const projectionGLSL = `
uniform vec2 u_angles; uniform float u_scale; uniform float u_aspect; uniform vec2 u_pan;
vec4 project(vec3 p){
 float cy=cos(u_angles.x),sy=sin(u_angles.x),cp=cos(u_angles.y),sp=sin(u_angles.y);
 vec3 q=vec3(cy*p.x+sy*p.z,p.y,-sy*p.x+cy*p.z);
 q=vec3(q.x,cp*q.y-sp*q.z,sp*q.y+cp*q.z);
 float perspective=4.5/(4.5-q.z);
 return vec4(q.x*perspective*u_scale/u_aspect+u_pan.x,q.y*perspective*u_scale+u_pan.y,-q.z/10.0,1.0);
}`;
class AtlasRenderer {
  constructor(canvas) {
    this.canvas=canvas;this.pending=false;this.lastFrame=performance.now();this.gridSize=66;this.density=new Float32Array(pointYears.length);this.terrain=[];this.ground=[];this.labels=[];
    this.gl=canvas.getContext('webgl',{alpha:true,antialias:true,preserveDrawingBuffer:true,powerPreference:'low-power'});
    this.fallback=!this.gl;
    if(this.fallback){this.ctx=canvas.getContext('2d');state.mode='map';resetCamera(false);Object.assign(camera,targetCamera);$('gesture-caption').textContent='2D fallback · drag to pan';document.querySelectorAll('[data-view]').forEach((b)=>{if(b.dataset.view!=='map'){b.disabled=true;b.title='Requires WebGL. All topic and time statistics are available in 2D.';}});}
    else this.initGL();
    this.makeLabels();this.updateDensity();this.resize();this.bindNavigation();
    new ResizeObserver(()=>this.resize()).observe($('scene'));
  }
  program(vertex,fragment){
    const gl=this.gl,p=gl.createProgram();
    for(const [type,source]of[[gl.VERTEX_SHADER,vertex],[gl.FRAGMENT_SHADER,fragment]]){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));gl.attachShader(p,s);}
    gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));return p;
  }
  initGL(){
    const gl=this.gl;
    this.pointsProgram=this.program(`precision highp float;attribute vec3 a_pos;attribute vec3 a_color;attribute float a_id;attribute float a_year;attribute float a_density;uniform float u_mode;uniform vec2 u_window;uniform float u_selected;uniform float u_compare;uniform float u_dpr;uniform float u_zoom;uniform float u_detail;varying vec3 v_color;varying float v_alpha;${projectionGLSL}
    void main(){vec3 p=a_pos;p.y=u_mode<.5?a_pos.y*1.65-.65:u_mode<1.5?a_density*.85-.65:-.15;gl_Position=project(p);float active=step(u_window.x,a_year)*step(a_year,u_window.y);float prior=(1.0-step(u_window.x,a_year))*step(a_year,u_window.y)*u_compare;float focus=u_selected<0.0?1.0:abs(a_id-u_selected)<.1?1.0:.22;v_alpha=(active*.56+prior*.12)*focus*u_detail;v_color=prior>.5?vec3(.36,.48,.51):a_color;float selected=u_selected>=0.0&&abs(a_id-u_selected)<.1?1.0:0.0;gl_PointSize=(1.25+.6*u_detail+selected*.65)*u_dpr*sqrt(u_zoom);if(active+prior<.5)gl_Position=vec4(2.,2.,2.,1.);}`,
    `precision mediump float;varying vec3 v_color;varying float v_alpha;void main(){float d=length(gl_PointCoord-vec2(.5))*2.;if(d>1.)discard;float glow=exp(-d*d*3.8);gl_FragColor=vec4(v_color,v_alpha*glow);}`);
    this.flatProgram=this.program(`precision highp float;attribute vec3 a_pos;attribute vec4 a_color;varying vec4 v_color;${projectionGLSL}void main(){v_color=a_color;gl_Position=project(a_pos);}`,
    `precision mediump float;varying vec4 v_color;void main(){gl_FragColor=v_color;}`);
    this.pointBuffer=gl.createBuffer();this.flatBuffer=gl.createBuffer();
    gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.disable(gl.DEPTH_TEST);
    this.canvas.addEventListener('webglcontextlost',(e)=>{e.preventDefault();stopPlayback();notify('The graphics context was interrupted. Reload to restore the map. Topic statistics remain available.');});
    this.canvas.addEventListener('webglcontextrestored',()=>{this.initGL();this.uploadPoints();this.request();});
  }
  uploadPoints(){
    if(this.fallback)return;const gl=this.gl,a=new Float32Array(pointYears.length*9);
    for(let i=0;i<pointYears.length;i++){const k=i*9;a[k]=positions[i*2];a[k+1]=pointYears[i]/yearMax;a[k+2]=positions[i*2+1];a.set(colors.subarray(i*3,i*3+3),k+3);a[k+6]=pointClusters[i];a[k+7]=pointYears[i];a[k+8]=this.density[i];}
    gl.bindBuffer(gl.ARRAY_BUFFER,this.pointBuffer);gl.bufferData(gl.ARRAY_BUFFER,a,gl.STATIC_DRAW);
  }
  updateDensity(){
    const N=this.gridSize,grid=new Float32Array(N*N),counts=new Float32Array(N*N*3);
    for(let i=0;i<pointYears.length;i++){if(pointYears[i]>state.to||pointYears[i]<state.from)continue;const x=clamp(Math.round((positions[i*2]/this.extentX+1)*.5*(N-1)),0,N-1),z=clamp(Math.round((positions[i*2+1]+1)*.5*(N-1)),0,N-1),j=z*N+x;grid[j]++;for(let c=0;c<3;c++)counts[j*3+c]+=colors[i*3+c];}
    let smooth=grid;
    for(let pass=0;pass<3;pass++){const next=new Float32Array(N*N);for(let z=0;z<N;z++)for(let x=0;x<N;x++){let v=0,w=0;for(let dz=-1;dz<=1;dz++)for(let dx=-1;dx<=1;dx++){const xx=x+dx,zz=z+dz;if(xx<0||zz<0||xx>=N||zz>=N)continue;const weight=dx===0&&dz===0?4:dx===0||dz===0?2:1;v+=smooth[zz*N+xx]*weight;w+=weight;}next[z*N+x]=v/w;}smooth=next;}
    const peak=Math.max(...smooth)||1;
    this.heightGrid=Float32Array.from(smooth,(v)=>Math.log1p(v)/Math.log1p(peak));
    for(let i=0;i<pointYears.length;i++){const x=clamp((positions[i*2]/this.extentX+1)*.5*(N-1),0,N-1),z=clamp((positions[i*2+1]+1)*.5*(N-1),0,N-1),x0=Math.floor(x),z0=Math.floor(z),x1=Math.min(N-1,x0+1),z1=Math.min(N-1,z0+1),tx=x-x0,tz=z-z0;this.density[i]=(1-tz)*((1-tx)*this.heightGrid[z0*N+x0]+tx*this.heightGrid[z0*N+x1])+tz*((1-tx)*this.heightGrid[z1*N+x0]+tx*this.heightGrid[z1*N+x1]);}
    const terrain=[],wire=[];
    const vertex=(x,z,alpha)=>{const j=z*N+x,h=this.heightGrid[j];return[(x/(N-1)*2-1)*this.extentX,h*.85-.65,z/(N-1)*2-1,.12+h*.14,.4+h*.22,.43+h*.12,alpha*(.13+h*.55)];};
    for(let z=0;z<N-1;z++)for(let x=0;x<N-1;x++){for(const [dx,dz]of[[0,0],[1,0],[0,1],[1,0],[1,1],[0,1]])terrain.push(...vertex(x+dx,z+dz,.18));if(x%2===0){wire.push(...vertex(x,z,.22),...vertex(x,z+1,.22));}if(z%2===0){wire.push(...vertex(x,z,.22),...vertex(x+1,z,.22));}}
    this.terrain=new Float32Array(terrain);this.wire=new Float32Array(wire);this.uploadPoints();this.makeGround();
  }
  get extentX(){const b=manifest.sample.bbox;return(b[2]-b[0])/(b[3]-b[1]);}
  makeGround(){
    const a=[],E=this.extentX+.05,levels=state.mode==='space'?[0,.5,1]:[0];
    const add=(p,q,alpha=.12)=>a.push(...p,.19,.45,.48,alpha,...q,.19,.45,.48,alpha);
    for(const t of levels){const y=state.mode==='map'?-.16:t*1.65-.65;add([-E,y,-1.05],[E,y,-1.05],.25);add([E,y,-1.05],[E,y,1.05],.25);add([E,y,1.05],[-E,y,1.05],.25);add([-E,y,1.05],[-E,y,-1.05],.25);if(t===0)for(let k=-1;k<=1.001;k+=.2){add([-E,y,k],[E,y,k],.12);add([k*E,y,-1.05],[k*E,y,1.05],.12);}}
    if(state.mode==='space')for(const x of[-E,E])for(const z of[-1.05,1.05])add([x,-.65,z],[x,1,z],.22);
    this.ground=new Float32Array(a);
  }
  resize(){
    const bounds=$('scene').getBoundingClientRect();this.width=bounds.width;this.height=bounds.height;this.dpr=Math.min(devicePixelRatio||1,1.8);this.canvas.width=Math.round(this.width*this.dpr);this.canvas.height=Math.round(this.height*this.dpr);
    const desktop=this.width>1000,left=desktop?(this.width>1600?315:this.width>1200?280:240):18,right=desktop?(this.width>1600?375:this.width>1200?330:290):40,top=desktop?105:this.width<620?225:155,bottom=desktop?230:220;
    this.viewport={x:left,y:top,width:Math.max(100,this.width-left-right),height:Math.max(100,this.height-top-bottom)};this.aspect=this.viewport.width/this.viewport.height;this.scale=Math.min(state.mode==='space'?.70:.88,this.aspect*.65);this.request();
  }
  project(x,y,z){
    const cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch),xx=cy*x+sy*z,zz=-sy*x+cy*z,yy=cp*y-sp*zz,depth=sp*y+cp*zz,p=4.5/(4.5-depth),s=this.scale*camera.zoom;
    return[this.viewport.x+(xx*p*s/this.aspect+camera.panX+1)*this.viewport.width/2,this.viewport.y+(1-(yy*p*s+camera.panY))*this.viewport.height/2,depth];
  }
  pointHeight(i){return state.mode==='space'?pointYears[i]/yearMax*1.65-.65:state.mode==='terrain'?this.density[i]*.85-.65:-.15;}
  request(){if(this.pending)return;this.pending=true;requestAnimationFrame(()=>this.render());}
  uniforms(program){const gl=this.gl;gl.uniform2f(gl.getUniformLocation(program,'u_angles'),camera.yaw,camera.pitch);gl.uniform1f(gl.getUniformLocation(program,'u_scale'),this.scale*camera.zoom);gl.uniform1f(gl.getUniformLocation(program,'u_aspect'),this.aspect);gl.uniform2f(gl.getUniformLocation(program,'u_pan'),camera.panX,camera.panY);}
  flat(array,mode){const gl=this.gl,p=this.flatProgram;gl.useProgram(p);gl.bindBuffer(gl.ARRAY_BUFFER,this.flatBuffer);gl.bufferData(gl.ARRAY_BUFFER,array,gl.DYNAMIC_DRAW);const pos=gl.getAttribLocation(p,'a_pos'),color=gl.getAttribLocation(p,'a_color');gl.enableVertexAttribArray(pos);gl.vertexAttribPointer(pos,3,gl.FLOAT,false,28,0);gl.enableVertexAttribArray(color);gl.vertexAttribPointer(color,4,gl.FLOAT,false,28,12);this.uniforms(p);gl.drawArrays(mode,0,array.length/7);gl.disableVertexAttribArray(pos);gl.disableVertexAttribArray(color);}
  render(){
    this.pending=false;this.scale=Math.min(state.mode==='space'?.70:.88,this.aspect*.65);let moving=false;const now=performance.now(),ease=reducedMotion.matches?1:1-Math.exp(-Math.max(16,now-this.lastFrame)/85);this.lastFrame=now;
    for(const key of Object.keys(camera)){const delta=targetCamera[key]-camera[key];if(Math.abs(delta)>.0001){camera[key]+=delta*ease;moving=true;}else camera[key]=targetCamera[key];}
    if(this.fallback)this.render2D();else{
      const gl=this.gl;gl.viewport(0,0,this.canvas.width,this.canvas.height);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);const v=this.viewport;gl.viewport(Math.round(v.x*this.dpr),Math.round((this.height-v.y-v.height)*this.dpr),Math.round(v.width*this.dpr),Math.round(v.height*this.dpr));
      this.makeGround();this.flat(this.ground,gl.LINES);if(state.mode==='terrain'){this.flat(this.terrain,gl.TRIANGLES);this.flat(this.wire,gl.LINES);}
      const p=this.pointsProgram;gl.useProgram(p);gl.bindBuffer(gl.ARRAY_BUFFER,this.pointBuffer);
      for(const[name,size,offset]of[['a_pos',3,0],['a_color',3,12],['a_id',1,24],['a_year',1,28],['a_density',1,32]]){const loc=gl.getAttribLocation(p,name);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,size,gl.FLOAT,false,36,offset);}
      this.uniforms(p);gl.uniform1f(gl.getUniformLocation(p,'u_mode'),state.mode==='space'?0:state.mode==='terrain'?1:2);gl.uniform2f(gl.getUniformLocation(p,'u_window'),state.from,state.to);gl.uniform1f(gl.getUniformLocation(p,'u_selected'),state.selected??-1);gl.uniform1f(gl.getUniformLocation(p,'u_compare'),state.compare?1:0);gl.uniform1f(gl.getUniformLocation(p,'u_dpr'),this.dpr);gl.uniform1f(gl.getUniformLocation(p,'u_zoom'),camera.zoom);gl.uniform1f(gl.getUniformLocation(p,'u_detail'),clamp(this.viewport.width/700,.4,1));gl.drawArrays(gl.POINTS,0,pointYears.length);
      for(const name of['a_pos','a_color','a_id','a_year','a_density'])gl.disableVertexAttribArray(gl.getAttribLocation(p,name));
    }
    this.renderLabels();if(moving)this.request();
  }
  render2D(){
    const ctx=this.ctx;ctx.setTransform(this.dpr,0,0,this.dpr,0,0);ctx.clearRect(0,0,this.width,this.height);ctx.save();const v=this.viewport;ctx.beginPath();ctx.rect(v.x,v.y,v.width,v.height);ctx.clip();
    for(let i=0;i<pointYears.length;i+=2){const year=pointYears[i];if(year>state.to||(!state.compare&&year<state.from))continue;const[x,y]=this.project(positions[i*2],-.15,positions[i*2+1]);ctx.globalAlpha=(year<state.from?.1:.55)*(state.selected===null||pointClusters[i]===state.selected?1:.2);ctx.fillStyle=cssColor(pointClusters[i]);ctx.fillRect(x,y,1.6,1.6);}ctx.restore();ctx.globalAlpha=1;
  }
  makeLabels(){
    $('map-labels').replaceChildren();
    this.labels=regions.map((r)=>{const el=document.createElement('button');el.className='map-label';el.textContent=r.name;el.tabIndex=-1;el.addEventListener('click',()=>selectRegion(r.id));$('map-labels').append(el);return{r,el};});
  }
  renderLabels(){
    const v=this.viewport,occupied=[],ranked=[...this.labels].sort((a,b)=>(b.r.id===state.selected?1:0)-(a.r.id===state.selected?1:0)||b.r.n_papers-a.r.n_papers);let shown=0;
    for(const{r,el}of ranked){const selected=r.id===state.selected,c=r._center;const y=state.mode==='space'?((state.from+state.to)/2/yearMax)*1.65-.65:state.mode==='terrain'?(r._density||.35)*.85-.65:-.15;const[x,yy]=this.project(c[0],y,c[1]);const max=innerWidth<620?3:6;const collision=occupied.some(([a,b])=>Math.abs(x-a)<125&&Math.abs(yy-b)<42);const visible=(selected||shown<max&&!collision)&&x>v.x+45&&x<v.x+v.width-45&&yy>v.y+15&&yy<v.y+v.height-15;el.hidden=!visible;if(visible){el.style.left=`${x}px`;el.style.top=`${yy-12}px`;el.classList.toggle('selected',selected);occupied.push([x,yy]);shown++;}}
    $('axis-labels').innerHTML='';
    if(state.mode==='space'){for(const t of[0,.5,1]){const[x,y]=this.project(-this.extentX-.04,t*1.65-.65,1.05);if(x<v.x+25||x>v.x+v.width-15||y<v.y+12||y>v.y+v.height-12)continue;const el=document.createElement('span');el.className='axis-label';el.textContent=String(Math.round(year0+t*yearMax));el.style.left=`${x-13}px`;el.style.top=`${y}px`;$('axis-labels').append(el);}}
  }
  pick(x,y){
    const v=this.viewport;if(x<v.x||x>v.x+v.width||y<v.y||y>v.y+v.height)return null;
    let best=null,distance=18*18;
    for(let i=0;i<pointYears.length;i+=3){if(pointYears[i]>state.to||(!state.compare&&pointYears[i]<state.from))continue;const[px,py]=this.project(positions[i*2],this.pointHeight(i),positions[i*2+1]),d=(px-x)**2+(py-y)**2;if(d<distance){distance=d;best={id:pointClusters[i],year:pointYears[i]+year0};}}
    return best;
  }
  bindNavigation(){
    const c=this.canvas,pointers=new Map();let drag=null,lastHover=0,pinch=0;
    c.addEventListener('contextmenu',(e)=>e.preventDefault());
    c.addEventListener('pointerdown',(e)=>{pointers.set(e.pointerId,[e.clientX,e.clientY]);c.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,pan:e.shiftKey||e.button===2};if(pointers.size===2){const a=[...pointers.values()];pinch=Math.hypot(a[0][0]-a[1][0],a[0][1]-a[1][1]);}$('map-tooltip').hidden=true;});
    c.addEventListener('pointermove',(e)=>{
      if(pointers.has(e.pointerId))pointers.set(e.pointerId,[e.clientX,e.clientY]);
      if(pointers.size===2){const a=[...pointers.values()],next=Math.hypot(a[0][0]-a[1][0],a[0][1]-a[1][1]);if(pinch>0)targetCamera.zoom=clamp(targetCamera.zoom*next/pinch,.45,5);pinch=next;this.request();return;}
      if(drag&&pointers.has(e.pointerId)){const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;if(drag.pan||state.mode==='map'){targetCamera.panX+=dx/this.viewport.width*2;targetCamera.panY-=dy/this.viewport.height*2;}else{targetCamera.yaw+=dx*.006;targetCamera.pitch=clamp(targetCamera.pitch+dy*.005,.08,1.56);}this.request();return;}
      if(e.pointerType==='touch'||performance.now()-lastHover<85)return;lastHover=performance.now();const bounds=c.getBoundingClientRect(),x=e.clientX-bounds.left,y=e.clientY-bounds.top,hit=this.pick(x,y),tip=$('map-tooltip');tip.hidden=!hit;if(hit){tip.innerHTML=`${esc(byId.get(hit.id)?.name||'Unassigned')}<small>Sampled paper · ${hit.year} · click to explore territory</small>`;tip.style.left=`${clamp(x+14,8,this.width-250)}px`;tip.style.top=`${Math.min(y+15,this.height-70)}px`;}
    });
    const finish=(e)=>{if(drag&&pointers.size===1&&Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)<5&&e.type!=='pointercancel'){const bounds=c.getBoundingClientRect(),hit=this.pick(e.clientX-bounds.left,e.clientY-bounds.top);if(hit&&byId.has(hit.id))selectRegion(hit.id);}pointers.delete(e.pointerId);drag=null;pinch=0;writeURL();};
    c.addEventListener('pointerup',finish);c.addEventListener('pointercancel',finish);c.addEventListener('pointerleave',()=>$('map-tooltip').hidden=true);
    c.addEventListener('wheel',(e)=>{e.preventDefault();targetCamera.zoom=clamp(targetCamera.zoom*Math.exp(-e.deltaY*.001),.45,5);this.request();writeURL();},{passive:false});
    c.addEventListener('keydown',(e)=>{if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-','Home',' '].includes(e.key))e.preventDefault();if(e.key==='ArrowLeft')targetCamera.yaw-=.1;if(e.key==='ArrowRight')targetCamera.yaw+=.1;if(e.key==='ArrowUp')targetCamera.pitch=clamp(targetCamera.pitch+.08,.08,1.57);if(e.key==='ArrowDown')targetCamera.pitch=clamp(targetCamera.pitch-.08,.08,1.57);if(['+','='].includes(e.key))targetCamera.zoom=clamp(targetCamera.zoom*1.15,.45,5);if(e.key==='-')targetCamera.zoom=clamp(targetCamera.zoom/1.15,.45,5);if(e.key==='Home')resetCamera();if(e.key===' ')togglePlayback();this.request();writeURL();});
  }
}

function stopPlayback(){state.playing=false;clearInterval(timer);$('play-icon').classList.remove('paused');$('play').setAttribute('aria-label','Play publication timeline');}
function togglePlayback(){
  if(state.playing){stopPlayback();return;}state.playing=true;$('play-icon').classList.add('paused');$('play').setAttribute('aria-label','Pause publication timeline');if(state.to>=yearMax)state.to=state.from;
  updateWindow();timer=setInterval(()=>{if(state.to<yearMax){state.to++;updateWindow();}else stopPlayback();},Number($('playback-speed').value));
}
function updateWindow(){updateControls();updateDetails();if($('topic-sort').value==='change')updateTopicList();renderer.updateDensity();renderer.request();writeURL();}
function story(step){
  if(step>3){closeStory();return;}state.tour=clamp(step,0,3);stopPlayback();$('story-card').hidden=false;$('topic-browser').classList.toggle('collapsed',innerWidth<1000);$('browser-toggle').setAttribute('aria-expanded',String(innerWidth>=1000));
  const largest=[...regions].sort((a,b)=>b.n_papers-a.n_papers)[0];const growing=[...regions].sort((a,b)=>(yearShare(b,year0+yearMax)-yearShare(b,year0+11))-(yearShare(a,year0+yearMax)-yearShare(a,year0+11)))[0];
  const steps=[
    ['Ideas have a landscape.',`${integer.format(manifest.sample.n_points)} real sampled papers, arranged by their semantic embeddings. Height adds publication year; orbit the map to see time unfold.`,'space',null,0,yearMax,false],
    ['Find a neighborhood.',`${largest.name} contains ${integer.format(largest.n_papers)} corpus papers. In 2D, related work becomes a neighborhood you can inspect, search, and revisit.`,'map',largest.id,0,yearMax,false],
    ['Look for a changing footprint.',`Compare ${year0+11}–${year0+yearMax}: earlier papers become a muted baseline. Density reveals concentration in the selected sample; the chart and counts use the monthly aggregate data.`,'terrain',growing.id,11,yearMax,true],
    ['Follow evidence into the future.','The solid line is observed smoothed share. The dashed line is an experimental forecast from June 2026. Its selection procedure missed the frozen criterion at all three historical cutoffs; read the methods before using it.','space',growing.id,0,yearMax,false]
  ];
  const[title,description,mode,id,from,to,compare]=steps[state.tour];state.mode=renderer.fallback?'map':mode;state.selected=id;state.micro=null;state.from=from;state.to=to;state.compare=compare;$('subarea-content').hidden=true;$('subarea-toggle').setAttribute('aria-expanded','false');$('microregion-select').replaceChildren();resetCamera();updateTopicList();updateWindow();
  $('story-step').textContent=`0${state.tour+1} / 04`;$('story-title').textContent=title;$('story-description').textContent=description;$('story-previous').disabled=state.tour===0;$('story-next').innerHTML=(state.tour===3?'Explore on your own':'Next chapter')+' <svg class="inline-icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M2 8h12m-5-5 5 5-5 5"/></svg>';
}
function closeStory(){state.tour=-1;$('story-card').hidden=true;}
function wrapText(ctx,text,x,y,maxWidth,lineHeight){let line='';for(const word of text.split(' ')){const next=line?`${line} ${word}`:word;if(ctx.measureText(next).width>maxWidth&&line){ctx.fillText(line,x,y);line=word;y+=lineHeight;}else line=next;}if(line)ctx.fillText(line,x,y);return y;}
async function exportImage(){
  Object.assign(camera,targetCamera);renderer.render();
  const v=renderer.viewport,W=Math.round(Math.max(1000,Math.min(1800,v.width*1.5))),scale=W/v.width,plotH=Math.round(v.height*scale),H=plotH+190;
  const out=document.createElement('canvas');out.width=W;out.height=H;const ctx=out.getContext('2d');
  ctx.fillStyle='#080d16';ctx.fillRect(0,0,W,H);
  ctx.drawImage(renderer.canvas,v.x*renderer.dpr,v.y*renderer.dpr,v.width*renderer.dpr,v.height*renderer.dpr,0,78,W,plotH);
  ctx.fillStyle='#e5f2e9';ctx.font='26px Helvetica,Arial,sans-serif';ctx.fillText('brontide.  The research atlas',30,38);
  ctx.fillStyle='#a7edcf';ctx.font='12px monospace';ctx.textAlign='right';ctx.fillText(`${state.mode==='space'?'Space-time':state.mode==='terrain'?'Sample density':'2D map'} · ${year0+state.from}–${year0+state.to}${state.compare?' · compare':''}`,W-30,35);ctx.textAlign='left';
  for(const{r,el}of renderer.labels){if(el.hidden)continue;ctx.font='13px Helvetica,Arial,sans-serif';ctx.fillStyle=r.id===state.selected?'#c5fbe0':'#b7cbd2';const x=(parseFloat(el.style.left)-v.x)*scale,y=(parseFloat(el.style.top)-v.y)*scale+78;wrapText(ctx,r.name,x-65,y,175,16);}
  document.querySelectorAll('.axis-label').forEach((el)=>{ctx.font='11px monospace';ctx.fillStyle='#92b4aa';ctx.fillText(el.textContent,(parseFloat(el.style.left)-v.x)*scale,(parseFloat(el.style.top)-v.y)*scale+78);});
  ctx.fillStyle='#c7ddd4';ctx.font='17px Helvetica,Arial,sans-serif';ctx.fillText(state.selected===null?'The frozen arXiv research landscape':byId.get(state.selected).name,30,plotH+112);
  const visible=pointYears.reduce((n,y)=>n+(y<=state.to&&(state.compare||y>=state.from)?1:0),0);
  ctx.fillStyle='#78929e';ctx.font='11px monospace';wrapText(ctx,`${integer.format(visible)} visible sampled dots · corpus through ${manifest.frozen_result.corpus_end} · fixed full-corpus UMAP · ${state.mode==='space'?'height: publication year':state.mode==='terrain'?'height: relative sample density':'2D display projection'}`,30,plotH+138,W-60,16);
  ctx.fillStyle='#576e7c';ctx.font='10px monospace';ctx.fillText(`numberingeometry.github.io/brontide · ${manifest.build_id}`,30,H-20);
  out.toBlob((blob)=>{if(!blob){notify('Image export could not complete.');return;}const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`brontide-${state.mode}-${year0+state.from}-${year0+state.to}.png`;a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);notify('Saved your view as a PNG with data provenance.');},'image/png');
}
function bindUI(){
  document.querySelectorAll('[data-view]').forEach((b)=>b.addEventListener('click',()=>setView(b.dataset.view)));
  $('topic-search').addEventListener('input',updateTopicList);$('topic-sort').addEventListener('change',updateTopicList);
  $('topic-list').addEventListener('click',(e)=>{const b=e.target.closest('[data-region]');if(b){selectRegion(Number(b.dataset.region));if(innerWidth<1000)collapseBrowser(true);}});
  $('clear-selection').addEventListener('click',()=>selectRegion(null));
  $('year-from').addEventListener('input',()=>{stopPlayback();state.from=Number($('year-from').value);state.to=Math.max(state.from,state.to);updateWindow();});
  $('year-to').addEventListener('input',()=>{stopPlayback();state.to=Number($('year-to').value);state.from=Math.min(state.from,state.to);updateWindow();});
  $('compare-toggle').addEventListener('change',()=>{state.compare=$('compare-toggle').checked;updateWindow();});
  $('play').addEventListener('click',togglePlayback);$('playback-speed').addEventListener('change',()=>{if(state.playing){stopPlayback();togglePlayback();}});
  $('zoom-in').addEventListener('click',()=>{targetCamera.zoom=clamp(targetCamera.zoom*1.2,.45,5);renderer.request();writeURL();});$('zoom-out').addEventListener('click',()=>{targetCamera.zoom=clamp(targetCamera.zoom/1.2,.45,5);renderer.request();writeURL();});$('reset-camera').addEventListener('click',()=>resetCamera());
  $('browser-toggle').addEventListener('click',()=>collapseBrowser(!$('topic-browser').classList.contains('collapsed')));
  $('method-open').addEventListener('click',()=>$('method-dialog').showModal());$('evidence-open').addEventListener('click',()=>$('method-dialog').showModal());$('method-close').addEventListener('click',()=>$('method-dialog').close());$('method-dialog').addEventListener('click',(e)=>{if(e.target===$('method-dialog')){const b=e.target.getBoundingClientRect();if(e.clientX<b.left||e.clientX>b.right||e.clientY<b.top||e.clientY>b.bottom)e.target.close();}});
  $('subarea-toggle').addEventListener('click',()=>openMicroregions());$('microregion-select').addEventListener('change',()=>{state.micro=$('microregion-select').value===''?null:Number($('microregion-select').value);updateDetails();writeURL();});
  $('story-start').addEventListener('click',()=>story(0));$('story-next').addEventListener('click',()=>story(state.tour+1));$('story-previous').addEventListener('click',()=>story(state.tour-1));$('story-close').addEventListener('click',closeStory);
  $('share-view').addEventListener('click',async()=>{writeURL(true);try{await navigator.clipboard.writeText(location.href);notify('View link copied. It restores the topic, time window, view, and camera.');}catch{const input=document.createElement('textarea');input.value=location.href;document.body.append(input);input.select();const copied=document.execCommand('copy');input.remove();notify(copied?'View link copied.':'The address bar contains your saved view. Copy its URL to share.');}});
  $('export-image').addEventListener('click',exportImage);
  $('history-chart').addEventListener('pointermove',(e)=>{if(!chartSeries)return;const b=e.currentTarget.getBoundingClientRect(),x=(e.clientX-b.left)/b.width*500,s=chartSeries,i=clamp(Math.round((x-s.L)/s.width*(s.values.length-1)),0,s.values.length-1),g=$('chart-hover');if(!g)return;g.style.display='';const px=s.X(i),py=s.Y(s.values[i]),label=`${s.months[i]}: ${s.values[i].toFixed(s.unit==='%'?3:0)}${s.unit}${i>=s.observed?' (forecast)':''}`,boxWidth=label.length*7.2+12,tx=clamp(px-boxWidth/2,5,495-boxWidth);g.querySelector('line').setAttribute('x1',px);g.querySelector('line').setAttribute('x2',px);g.querySelector('circle').setAttribute('cx',px);g.querySelector('circle').setAttribute('cy',py);const rect=g.querySelector('rect');rect.setAttribute('x',tx);rect.setAttribute('width',boxWidth);const text=g.querySelector('text');text.setAttribute('x',tx+6);text.textContent=label;});$('history-chart').addEventListener('pointerleave',()=>{const g=$('chart-hover');if(g)g.style.display='none';});
  document.addEventListener('keydown',(e)=>{if(e.key==='/'&&!['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName)&&!$('method-dialog').open){e.preventDefault();collapseBrowser(false);$('topic-search').focus();}if(e.key==='Escape'&&!$('method-dialog').open){closeStory();selectRegion(null);}});
  window.addEventListener('hashchange',async()=>{stopPlayback();readURL();updateControls();updateTopicList();updateDetails();renderer.updateDensity();renderer.request();if(state.micro!==null)await openMicroregions(true);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stopPlayback();});
}
function collapseBrowser(collapsed){$('topic-browser').classList.toggle('collapsed',collapsed);$('browser-toggle').setAttribute('aria-expanded',String(!collapsed));$('browser-toggle').textContent=collapsed?'+':'−';}
async function init(){
  try{
    manifest=await getFile('manifest.json');year0=manifest.sample.year0;yearMax=manifest.sample.year1-year0;
    const[raw,clusters,years,regionData]=await Promise.all([getFile(manifest.files.points,true),getFile(manifest.files.cluster,true),getFile(manifest.files.year,true),getFile(manifest.files.regions)]);
    data=regionData;regions=data.regions;byId=new Map(regions.map((r)=>[r.id,r]));pointYears=new Uint8Array(years);pointClusters=new Uint8Array(clusters);
    if(pointYears.length!==manifest.sample.n_points||pointClusters.length!==pointYears.length||raw.byteLength!==pointYears.length*4)throw new Error('The map bundle has inconsistent point dimensions.');
    const xy=new DataView(raw),offset=manifest.sample.quantization.offset,b=manifest.sample.bbox,aspect=(b[2]-b[0])/(b[3]-b[1]);positions=new Float32Array(pointYears.length*2);colors=new Float32Array(pointYears.length*3);
    const centers=new Map(regions.map((r)=>[r.id,[0,0,0]]));
    for(let i=0;i<pointYears.length;i++){positions[i*2]=xy.getInt16(i*4,true)/offset*aspect;positions[i*2+1]=xy.getInt16(i*4+2,true)/offset;colors.set(regionColor(pointClusters[i]),i*3);const c=centers.get(pointClusters[i]);if(c){c[0]+=positions[i*2];c[1]+=positions[i*2+1];c[2]++;}}
    for(const r of regions){const c=centers.get(r.id);r._center=c[2]?[c[0]/c[2],c[1]/c[2]]:[0,0];}
    readURL();for(const id of['year-from','year-to'])$(id).max=yearMax;
    renderer=new AtlasRenderer($('atlas-canvas'));bindUI();if(innerWidth<=1000)collapseBrowser(true);
    $('corpus-size').textContent=compact.format(manifest.frozen_result.n_papers);$('territory-count').textContent=regions.length;
    const date=new Date(`${manifest.frozen_result.corpus_end}-01T00:00:00Z`).toLocaleDateString('en-US',{year:'numeric',month:'long',timeZone:'UTC'});$('footer-date').textContent=date;
    $('method-sample').textContent=`${integer.format(manifest.sample.n_points)} dots form a stratified display sample of ${integer.format(manifest.frozen_result.n_papers)} indexed papers, through ${date}. Monthly aggregates cover ${integer.format(manifest.monthly_series.n_papers)} records; the source artifacts do not identify ${integer.format(manifest.monthly_series.corpus_records_outside_series)} corpus records outside the monthly table. Window statistics use that table, not dot counts.`;
    $('build-note').textContent=`Build ${manifest.build_id} · ${manifest.geometry.embedding} · PCA ${manifest.geometry.pca_dimensions}D → ${manifest.geometry.display_projection} · frozen observations ${manifest.frozen_result.corpus_start}–${manifest.frozen_result.corpus_end}. No live API calls or per-paper identifiers in this map.`;
    updateControls();updateTopicList();updateDetails();renderer.request();$('loading').hidden=true;
    if(state.micro!==null)await openMicroregions(true);
    if(renderer.fallback)notify('WebGL is unavailable. The 2D atlas preserves search, time controls, and all statistics.');
    window.__BRONTIDE__=Object.freeze({getState:()=>({...state,camera:{...targetCamera},webgl:!renderer.fallback}),getData:()=>({manifest,regions:data.regions,months:data.months,monthly_totals:data.monthly_totals,microregions:microData?.regions}),getVisiblePointCount:()=>pointYears.reduce((n,y)=>n+(y<=state.to&&(state.compare||y>=state.from)?1:0),0)});
  }catch(error){console.error('Atlas initialization failed:',error);$('loading').classList.add('error');$('loading').innerHTML=`<span class="eyebrow">THE ATLAS COULD NOT OPEN</span><p>${esc(error.message)} Check your connection and retry, or explore the static map and reproducible source.</p><div><button id="retry-load">Try again</button><a href="figures/frontier-map.png">Static map ↗</a></div>`;$('retry-load').addEventListener('click',()=>location.reload());}
}
init();
