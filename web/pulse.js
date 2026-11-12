/* Pulse view: how far the mix of research turned, year over year.
   Every drawn value comes from artifacts/explorer/pulse.json; nothing is fitted here. */
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const fmtMonth = (m) => `${MONTHS[Number(m.slice(5, 7)) - 1]} ${m.slice(0, 4)}`;
const shift = (m, k) => { const i = Number(m.slice(0, 4)) * 12 + Number(m.slice(5, 7)) - 1 + k; return `${Math.floor(i / 12)}-${String(i % 12 + 1).padStart(2, '0')}`; };
const deg = (v) => `${v.toFixed(1)}°`;
const short = (name, n = 30) => (name.length > n ? `${name.slice(0, n - 1).trimEnd()}…` : name);

export class PulseView {
  constructor(root, ctx) {
    this.root = root; this.ctx = ctx; this.data = null; this.month = null; this.hover = null; this.pending = false; this.pointer = null; this.valueIndex = null;
    this.chart = root.querySelector('#pulse-chart'); this.compass = root.querySelector('#pulse-compass');
    this.movers = root.querySelector('#pulse-movers');
    this.chart.addEventListener('pointermove', (e) => {
      // Ignore synthetic moves at an unchanged position, so a key press keeps its month.
      if (!this.data || (this.pointer && this.pointer[0] === e.clientX && this.pointer[1] === e.clientY)) return;
      this.pointer = [e.clientX, e.clientY]; this.hover = this.indexAt(e); this.request();
    });
    this.chart.addEventListener('pointerleave', () => { this.pointer = null; this.hover = null; this.request(); });
    this.chart.addEventListener('click', (e) => { if (!this.data) return; const i = this.indexAt(e); this.hover = null; this.pin(i); });
    this.chart.addEventListener('keydown', (e) => {
      if (!this.data) return;
      this.hover = null;
      const n = this.data.months.length, i = this.month ?? this.dockIndex();
      const next = { ArrowLeft: i - 1, ArrowRight: i + 1, PageUp: i - 12, PageDown: i + 12, Home: 0, End: n - 1 }[e.key];
      if (next !== undefined) { e.preventDefault(); this.pin(Math.max(0, Math.min(n - 1, next))); }
      if (e.key === 'Enter') { const lead = this.data.territories.leaders[i][0]; if (lead) ctx.select(lead[0]); }
    });
    this.movers.addEventListener('click', (e) => {
      if (e.target.closest('[data-retry]')) { this.retry(); return; }
      const b = e.target.closest('[data-region]'); if (b) ctx.select(Number(b.dataset.region));
    });
    const watch = new ResizeObserver(() => this.request()); [root, this.chart, this.compass].forEach((el) => watch.observe(el));
  }
  async load() {
    if (!this.data) { this.data = await this.ctx.load(); this.chart.setAttribute('aria-valuemax', String(this.data.months.length - 1)); }
    return this.data;
  }
  fail() {
    this.chart.removeAttribute('viewBox');
    this.chart.innerHTML = `<text x="34" y="40" font-size="13" fill="${this.ctx.theme().muted}">The pulse data could not load.</text>`;
    this.movers.innerHTML = '<button data-retry type="button">Try again</button>';
  }
  async retry() { this.movers.innerHTML = ''; try { await this.load(); this.request(); } catch (error) { console.error(error); this.fail(); } }
  indexAt(e) {
    const box = this.chart.getBoundingClientRect(), L = this.layout, n = this.data.months.length;
    if (!L) return this.month ?? this.dockIndex();
    const x = (e.clientX - box.left) / box.width * L.w;
    return Math.max(0, Math.min(n - 1, Math.round((x - L.left) / L.pw * (n - 1))));
  }
  dockIndex() {
    const months = this.data.months, last = this.ctx.lastYear();
    const i = months.findLastIndex((m) => Number(m.slice(0, 4)) <= last);
    return i < 0 ? 0 : i;
  }
  current() { return this.hover ?? this.month ?? this.dockIndex(); }
  pin(i) { this.month = i; this.ctx.onPin(this.data.months[i]); this.request(); }
  /** Pin a month by its label; returns false (and clears the pin) when the month is unknown. */
  setMonth(m) { const i = m ? this.data?.months.indexOf(m) ?? -1 : -1; this.month = i >= 0 ? i : null; this.request(); return i >= 0; }
  clearPin() { this.month = null; this.request(); }
  windowText(i) { const m = this.data.months[i]; return `${fmtMonth(shift(m, -11))} – ${fmtMonth(m)}`; }
  readout(i) {
    const d = this.data;
    return `${this.windowText(i)}: ${deg(d.territories.angle_deg[i])} across 64 fields, ${deg(d.microregions.angle_deg[i])} including 1,024 subtopics, ${deg(d.within.angle_deg[i])} inside fields`;
  }
  request() { if (this.pending) return; this.pending = true; requestAnimationFrame(() => { this.pending = false; if (this.ctx.active() && this.data) this.render(); }); }
  render() {
    const t = this.ctx.theme(), i = this.current();
    this.renderMovers(t, i); this.renderChart(t, i); this.renderCompass(t, i);
    // The slider value is the pinned or dock month; hovering does not change it.
    const v = this.month ?? this.dockIndex();
    if (v !== this.valueIndex) {
      this.valueIndex = v;
      this.chart.setAttribute('aria-valuenow', String(v));
      this.chart.setAttribute('aria-valuetext', this.readout(v));
    }
  }
  renderChart(t, i) {
    const box = this.chart.getBoundingClientRect(), w = Math.max(260, box.width), h = Math.max(160, box.height);
    const { svg, layout } = this.chartMarkup(t, i, w, h);
    this.layout = layout; this.chart.setAttribute('viewBox', `0 0 ${w} ${h}`); this.chart.innerHTML = svg;
  }
  /** Chart markup for a given size; the export calls this with a fixed desktop size. */
  chartMarkup(t, i, w, h) {
    const d = this.data, esc = this.ctx.esc, narrow = w < 560, left = 34, right = 10, bottom = 24, pw = w - left - right;
    const n = d.months.length, A = d.territories.angle_deg, B = d.microregions.angle_deg, F = d.microregions.floor_angle_deg, Wn = d.within.angle_deg;
    const X = (k) => left + k / (n - 1) * pw;
    // Peak labels sit in up to two rows above the plot; a label that fits in neither row is left out.
    const head = narrow ? 70 : 54, rows = [[], []], pills = [];
    for (const p of d.highlights) {
      const label = narrow ? p.month.slice(0, 4) : `${p.month.slice(0, 4)} · ${short(this.ctx.name(p.leader), 24)}`;
      const tw = label.length * 5.6 + 16, tx = Math.max(left, Math.min(w - right - tw, X(p.index) - tw / 2));
      const row = rows.findIndex((r) => r.every(([a, b]) => tx + tw + 6 < a || tx > b + 6));
      if (row < 0) continue;
      rows[row].push([tx, tx + tw]); pills.push({ p, label, tw, tx, row });
    }
    const used = Math.max(1, ...pills.map((q) => q.row + 1)), top = head + 20 * used + 8, ph = Math.max(40, h - top - bottom);
    const ymax = Math.ceil(Math.max(...B, ...A) / 2) * 2, Y = (v) => top + ph - v / ymax * ph;
    const line = (V) => V.map((v, k) => `${k ? 'L' : 'M'}${X(k).toFixed(1)},${Y(v).toFixed(1)}`).join('');
    const area = (V) => `${line(V)}L${X(n - 1).toFixed(1)},${Y(0)}L${X(0)},${Y(0)}Z`;
    const first = this.ctx.firstYear(), fi = Math.max(0, d.months.findIndex((m) => Number(m.slice(0, 4)) >= first)), li = this.dockIndex();
    let s = `<rect x="${X(fi)}" y="${top - 6}" width="${Math.max(0, X(li) - X(fi))}" height="${ph + 6}" rx="8" fill="${t.history}" opacity=".055"/>`;
    for (let g = 2; g <= ymax; g += 2) s += `<line x1="${left}" x2="${w - right}" y1="${Y(g)}" y2="${Y(g)}" stroke="${t.grid}" stroke-opacity=".16" stroke-dasharray="2 5"/><text x="${left - 7}" y="${Y(g) + 3}" text-anchor="end" font-size="10" fill="${t.muted}">${g}°</text>`;
    s += `<line x1="${left}" x2="${w - right}" y1="${Y(0)}" y2="${Y(0)}" stroke="${t.grid}" stroke-opacity=".35"/>`;
    s += `<path d="${area(B)}" fill="${t.history}" fill-opacity=".14"/><path d="${line(B)}" fill="none" stroke="${t.history}" stroke-opacity=".6" stroke-width="1.2" stroke-linejoin="round"/>`;
    s += `<path d="${area(A)}" fill="${t.history}" fill-opacity=".34"/><path d="${line(A)}" fill="none" stroke="${t.ink}" stroke-width="1.7" stroke-linejoin="round"/>`;
    s += `<path d="${line(F)}" fill="none" stroke="${t.muted}" stroke-opacity=".55" stroke-width="1" stroke-dasharray="1 4" stroke-linecap="round"/>`;
    const years = narrow ? [2009, 2014, 2019, 2024] : [2009, 2012, 2015, 2018, 2021, 2024];
    for (const y of years) { const k = d.months.indexOf(`${y}-01`); if (k >= 0) s += `<text x="${X(k)}" y="${h - 6}" text-anchor="middle" font-size="10" fill="${t.muted}">${y}</text>`; }
    s += `<text x="${X(n - 1)}" y="${h - 6}" text-anchor="end" font-size="10" fill="${t.muted}">2026</text>`;
    // Orientation labels at prominent peaks, named by their largest contributor.
    for (const { p, label, tw, tx, row } of pills) {
      const x = X(p.index), y = Y(A[p.index]), ly = head + 10 + 20 * row;
      s += `<line x1="${x}" x2="${x}" y1="${ly + 8}" y2="${y - 4}" stroke="${t.gold}" stroke-opacity=".45" stroke-dasharray="1 3"/><circle cx="${x}" cy="${y}" r="3" fill="${t.paper}" stroke="${t.gold}" stroke-width="1.4"/>`;
      s += `<rect x="${tx}" y="${ly - 9}" width="${tw}" height="17" rx="8.5" fill="${t.paper}" stroke="${t.gold}" stroke-opacity=".5"/><text x="${tx + tw / 2}" y="${ly + 3}" text-anchor="middle" font-size="9.5" font-weight="700" fill="${t.gold}">${esc(label)}</text>`;
    }
    // Cursor and readout.
    const cx = X(i);
    s += `<line x1="${cx}" x2="${cx}" y1="${top - 4}" y2="${Y(0)}" stroke="${t.ink}" stroke-opacity=".55" stroke-dasharray="2 4"/>`;
    s += `<circle cx="${cx}" cy="${Y(B[i])}" r="3.5" fill="${t.paper}" stroke="${t.history}" stroke-width="1.6"/><circle cx="${cx}" cy="${Y(A[i])}" r="4" fill="${t.ink}"/>`;
    s += `<text x="${left}" y="13" font-size="9" font-weight="700" letter-spacing="1" fill="${t.muted}">${narrow ? 'HOW FAR THE MIX TURNED' : 'HOW FAR THE MIX TURNED · EACH YEAR VS THE YEAR BEFORE'}</text>`;
    s += `<g font-size="10.5" fill="${t.ink}"><rect x="${left}" y="22" width="13" height="8" rx="3" fill="${t.history}" fill-opacity=".55" stroke="${t.ink}" stroke-width="1"/><text x="${left + 18}" y="30"><tspan font-weight="700">${deg(A[i])}</tspan> across 64 fields</text>`;
    const second = left + (narrow ? 0 : 150), secondY = narrow ? 46 : 30, thirdY = narrow ? 62 : 46;
    s += `<rect x="${second}" y="${secondY - 8}" width="13" height="8" rx="3" fill="${t.history}" fill-opacity=".2" stroke="${t.history}" stroke-opacity=".7"/><text x="${second + 18}" y="${secondY}"><tspan font-weight="700">${deg(B[i])}</tspan> including 1,024 subtopics</text>`;
    s += `<text x="${left + 18}" y="${thirdY}" fill="${t.muted}"><tspan font-weight="700" fill="${t.ink}">${deg(Wn[i])}</tspan> turned inside fields</text></g>`;
    s += `<text x="${w - right}" y="30" text-anchor="end" font-size="10.5" font-weight="700" fill="${t.ink}">${this.windowText(i)}</text>`;
    s += `<text x="${w - right}" y="${thirdY}" text-anchor="end" font-size="9" fill="${t.muted}">${narrow ? 'dotted: counting alone' : 'dotted: apparent turn from counting alone'}</text>`;
    return { svg: s, layout: { w, h, left, pw } };
  }
  renderCompass(t, i) {
    const box = this.compass.getBoundingClientRect(); if (box.width < 40) return;
    const w = box.width, h = Math.max(120, box.height);
    this.compass.setAttribute('viewBox', `0 0 ${w} ${h}`); this.compass.innerHTML = this.compassMarkup(t, i, w, h);
    const T = this.data.trajectory, ax = T.axes[0], full = (ids) => ids.slice(0, 2).map((id) => this.ctx.name(id)).join(' and ');
    this.compass.setAttribute('aria-label', `Trajectory of arXiv's yearly field mix from ${T.years[0]} to ${T.years.at(-1)}, projected on its two main directions of change. Total turn ${deg(T.direct_angle_deg)}; summed yearly steps ${deg(T.path_angle_deg)}. Toward ${full(ax.positive)}; away from ${full(ax.negative)}.`);
  }
  compassMarkup(t, i, w, h) {
    const T = this.data.trajectory, pad = 26, top = 40, bottom = 46, esc = this.ctx.esc;
    const xs = T.x, ys = T.y, minx = Math.min(...xs), maxx = Math.max(...xs), miny = Math.min(...ys), maxy = Math.max(...ys);
    const scale = Math.min((w - 2 * pad) / (maxx - minx), (h - top - bottom) / (maxy - miny)), ox = (w - (maxx - minx) * scale) / 2, oy = top + (h - top - bottom - (maxy - miny) * scale) / 2;
    const P = xs.map((x, k) => [ox + (x - minx) * scale, oy + (maxy - ys[k]) * scale]);
    let d = `M${P[0][0].toFixed(1)},${P[0][1].toFixed(1)}`;
    for (let k = 0; k < P.length - 1; k++) {
      const p0 = P[Math.max(0, k - 1)], p1 = P[k], p2 = P[k + 1], p3 = P[Math.min(P.length - 1, k + 2)];
      d += `C${(p1[0] + (p2[0] - p0[0]) / 6).toFixed(1)},${(p1[1] + (p2[1] - p0[1]) / 6).toFixed(1)} ${(p2[0] - (p3[0] - p1[0]) / 6).toFixed(1)},${(p2[1] - (p3[1] - p1[1]) / 6).toFixed(1)} ${p2[0].toFixed(1)},${p2[1].toFixed(1)}`;
    }
    const year = Math.max(T.years[0], Math.min(T.years.at(-1), Number(this.data.months[i].slice(0, 4)))), yi = T.years.indexOf(year);
    let s = `<text x="0" y="13" font-size="9" font-weight="700" letter-spacing="1" fill="${t.muted}">THE LONG TURN · ${T.years[0]}–${T.years.at(-1)}</text>`;
    s += `<text x="0" y="30" font-size="10.5" fill="${t.ink}">The mix turned <tspan font-weight="700">${deg(T.direct_angle_deg)}</tspan> along a ${deg(T.path_angle_deg)} yearly path</text>`;
    s += `<path d="${d}" fill="none" stroke="${t.history}" stroke-opacity=".28" stroke-width="7" stroke-linecap="round"/><path d="${d}" fill="none" stroke="${t.history}" stroke-width="1.6" stroke-linecap="round"/>`;
    P.forEach(([x, y], k) => {
      const on = k === yi, label = k === 0 || k === P.length - 1 || on;
      s += `<circle cx="${x}" cy="${y}" r="${on ? 5 : 2.4}" fill="${on ? t.gold : t.paper}" stroke="${on ? t.paper : t.history}" stroke-width="${on ? 1.5 : 1.2}"/>`;
      if (label) s += `<text x="${x}" y="${y + (on ? 18 : -10)}" text-anchor="middle" font-size="${on ? 10.5 : 9.5}" font-weight="${on ? 700 : 400}" fill="${on ? t.gold : t.muted}">${T.years[k]}</text>`;
    });
    const ax = T.axes[0], names = (ids) => ids.slice(0, w < 400 ? 1 : 2).map((id) => short(this.ctx.name(id), w < 400 ? 22 : 18)).join(' · ');
    s += `<text x="0" y="${h - 22}" font-size="9.5" fill="${t.muted}">← ${esc(names(ax.negative))}</text><text x="${w}" y="${h - 22}" text-anchor="end" font-size="9.5" fill="${t.muted}">${esc(names(ax.positive))} →</text>`;
    s += `<text x="0" y="${h - 6}" font-size="8.5" fill="${t.quiet}">${w < 400 ? 'Flat view of a curved path' : 'A flat view of a curved path on the sphere of compositions'} · ${Math.round(100 * (T.explained[0] + T.explained[1]))}% of its spread</text>`;
    return s;
  }
  renderMovers(t, i) {
    const rows = this.data.territories.leaders[i].slice(0, 5), max = Math.max(...rows.map((r) => r[1]), 1);
    this.movers.innerHTML = `<div class="pulse-movers-head"><span>WHAT MOVED</span><small>${this.windowText(i)}</small></div>` + rows.map(([id, pct, dir]) => {
      const name = this.ctx.name(id), color = this.ctx.color(id);
      return `<button data-region="${id}" title="Open ${this.ctx.esc(name)}"><i style="background:${color}"></i><span class="pulse-name">${this.ctx.esc(name)}</span><span class="pulse-dir ${dir > 0 ? 'up' : 'down'}" aria-label="${dir > 0 ? 'rising share' : 'falling share'}">${dir > 0 ? '↑' : '↓'}</span><span class="pulse-bar"><b style="width:${Math.max(4, 100 * pct / max).toFixed(1)}%"></b></span><span class="pulse-pct">${pct.toFixed(0)}%</span></button>`;
    }).join('') + '<p>Share of the 64-field turn, on the 1 − cos scale. Arrows mark rising or falling share.</p>';
  }
}
