/**
 * Dependency-free SVG line chart: TypeScript port of the mockups' `shared/chart.js`.
 *
 * Same visuals and interactions (bands, per-run lines, dashes, left/right axes, log y, nice
 * ticks, crosshair tooltip, drag-zoom on x with reset, point click), plus:
 * - the right-axis label is drawn;
 * - hidden series are excluded from the domains;
 * - tooltips use binary search;
 * - hover state lives in its own layer, so moving the mouse never rebuilds the data paths;
 * - `exportSVG()` / `exportPNG()`;
 * - theme values come from CSS variables (`themeFromCSS`).
 *
 * `mount(container, spec)` → `{ update, redraw, resetZoom, exportSVG, exportPNG, destroy }`.
 * Resizing is the caller's job (see `LineChart.vue`): call `redraw()`.
 */
import { fmtTick } from "../domain/format";

const NS = "http://www.w3.org/2000/svg";

export type ChartAxis = "left" | "right";
export interface ChartRunInput {
  seed: number | null;
  label?: string;
  x: number[];
  y: (number | null)[];
}
export interface ChartSeriesInput {
  key?: string;
  label: string;
  color: string;
  dash?: string;
  width?: number;
  axis?: ChartAxis;
  x: number[];
  /** Centre line (may be empty: runs only). */
  y?: (number | null)[];
  lo?: (number | null)[] | null;
  hi?: (number | null)[] | null;
  runs?: ChartRunInput[];
  runOpacity?: number;
  runWidth?: number;
  bandOpacity?: number;
  hidden?: boolean;
}
export interface ChartTheme {
  text: string;
  grid: string;
  axis: string;
  font: string;
  tooltipBg: string;
  tooltipText: string;
  bg: string;
}
export interface ChartSpec {
  series: ChartSeriesInput[];
  theme?: Partial<ChartTheme>;
  xLabel?: string;
  yLabel?: string;
  yLabelRight?: string;
  logY?: boolean;
  compact?: boolean;
  emptyText?: string;
  /** Called with the nearest x of the first visible series and that series' index. */
  onPointClick?: ((x: number, seriesIndex: number) => void) | null;
  xFormat?: (v: number) => string;
}
export interface ChartHandle {
  update(spec: ChartSpec): void;
  redraw(): void;
  resetZoom(): void;
  /** Current zoomed x range, or null. */
  readonly zoom: [number, number] | null;
  exportSVG(): string;
  exportPNG(scale?: number): Promise<Blob>;
  destroy(): void;
}
export type MountOptions = {
  /** Fixed size (e.g. tests, export); by default the container's client size is used. */
  width?: number;
  height?: number;
};

export const DEFAULT_THEME: ChartTheme = {
  text: "#7a7a88",
  grid: "rgba(30,30,70,.06)",
  axis: "rgba(30,30,70,.22)",
  font: "11px system-ui, -apple-system, sans-serif",
  tooltipBg: "rgba(30,27,60,.94)",
  tooltipText: "#fff",
  bg: "#fff",
};

const THEME_VARS: Record<keyof ChartTheme, string> = {
  text: "--chart-text",
  grid: "--chart-grid",
  axis: "--chart-axis",
  font: "--chart-font",
  tooltipBg: "--chart-tooltip-bg",
  tooltipText: "--chart-tooltip-text",
  bg: "--chart-bg",
};

/** Read the chart theme from the CSS variables in scope of `el` (defaults when unset). @ai-generated */
export function themeFromCSS(el: Element): ChartTheme {
  const cs = getComputedStyle(el);
  const out = { ...DEFAULT_THEME };
  for (const k of Object.keys(THEME_VARS) as (keyof ChartTheme)[]) {
    const v = cs.getPropertyValue(THEME_VARS[k]).trim();
    if (v) out[k] = v;
  }
  return out;
}

// ---------------------------------------------------------------- pure helpers

/** "Nice" tick values (1/2/5 × 10^k) covering [min, max] with about `count` ticks. @ai-generated */
export function niceTicks(min: number, max: number, count: number): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const span = max - min;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / count)));
  const err = (count * step0) / span;
  const step = err <= 0.15 ? step0 * 10 : err <= 0.35 ? step0 * 5 : err <= 0.75 ? step0 * 2 : step0;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

/** Index of the element of sorted `xs` nearest to `v` (binary search); -1 when empty. @ai-generated */
export function nearestIndex(xs: ArrayLike<number>, v: number): number {
  const n = xs.length;
  if (!n) return -1;
  let lo = 0;
  let hi = n - 1;
  if (v <= xs[0]) return 0;
  if (v >= xs[hi]) return hi;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= v) lo = mid;
    else hi = mid;
  }
  return v - xs[lo] <= xs[hi] - v ? lo : hi;
}

const r1 = (v: number) => Math.round(v * 10) / 10;

/**
 * SVG path of a polyline; null/non-finite values (and ≤ 0 in log scale) lift the pen.
 *
 * @ai-generated
 */
export function buildPath(xs: ArrayLike<number>, ys: ArrayLike<number | null>, X: (v: number) => number, Y: (v: number) => number, logY = false): string {
  let d = "";
  let pen = false;
  const n = Math.min(xs.length, ys.length);
  for (let i = 0; i < n; i++) {
    const yv = ys[i];
    if (yv === null || !Number.isFinite(yv) || (logY && yv <= 0)) {
      pen = false;
      continue;
    }
    d += (pen ? "L" : "M") + r1(X(xs[i])) + "," + r1(Y(yv));
    pen = true;
  }
  return d;
}

/** Closed band path (hi forward, lo backward); gaps are skipped. @ai-generated */
export function buildBandPath(
  xs: ArrayLike<number>,
  lo: ArrayLike<number | null>,
  hi: ArrayLike<number | null>,
  X: (v: number) => number,
  Y: (v: number) => number,
  logY = false,
): string {
  const ok = (v: number | null): v is number => v !== null && Number.isFinite(v) && (!logY || v > 0);
  let d = "";
  for (let i = 0; i < xs.length; i++) {
    const v = hi[i];
    if (ok(v)) d += (d ? "L" : "M") + r1(X(xs[i])) + "," + r1(Y(v));
  }
  if (!d) return "";
  for (let i = xs.length - 1; i >= 0; i--) {
    const v = lo[i];
    if (ok(v)) d += "L" + r1(X(xs[i])) + "," + r1(Y(v));
  }
  return d + "Z";
}

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, parent?: Element): SVGElementTagNameMap[K] {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, String(attrs[k]));
  if (parent) parent.appendChild(e);
  return e;
}

const escapeHtml = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
const fmtVal = (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : String(+(+v).toPrecision(3)));

type Geometry = {
  W: number;
  H: number;
  m: { l: number; r: number; t: number; b: number };
  iw: number;
  ih: number;
  xmin: number;
  xmax: number;
  invX: (px: number) => number;
};

let clipSeq = 0;

// ---------------------------------------------------------------- mount

/**
 * Mount a chart into `container` (which gets `position: relative` for the tooltip).
 *
 * @ai-generated
 */
export function mount(container: HTMLElement, initial: ChartSpec, opts: MountOptions = {}): ChartHandle {
  if (!container.style.position) container.style.position = "relative";
  const svg = el("svg", { width: "100%", height: "100%", class: "mc-svg" });
  svg.style.display = "block";
  svg.style.userSelect = "none";
  container.appendChild(svg);
  const tip = document.createElement("div");
  tip.className = "mc-tip";
  Object.assign(tip.style, {
    position: "absolute",
    pointerEvents: "none",
    display: "none",
    padding: "6px 8px",
    borderRadius: "6px",
    fontSize: "11px",
    lineHeight: "1.45",
    zIndex: "5",
    whiteSpace: "nowrap",
    boxShadow: "0 4px 16px rgba(0,0,0,.18)",
  });
  container.appendChild(tip);

  const clipId = `mc-clip-${++clipSeq}`;
  let spec = initial;
  let xr: [number, number] | null = null;
  let geom: Geometry | null = null;
  let theme: ChartTheme = DEFAULT_THEME;
  let hover: { cross: SVGLineElement; sel: SVGRectElement } | null = null;
  let dragStart: number | null = null;

  const size = () => ({ W: opts.width ?? container.clientWidth, H: opts.height ?? container.clientHeight });

  function draw(): void {
    theme = { ...DEFAULT_THEME, ...(spec.theme ?? {}) };
    Object.assign(tip.style, { background: theme.tooltipBg, color: theme.tooltipText, font: theme.font });
    tip.style.display = "none";
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    geom = null;
    hover = null;
    const { W, H } = size();
    svg.setAttribute("data-width", String(W));
    svg.setAttribute("data-height", String(H));
    if (W < 40 || H < 40) return;
    const s = spec;
    const visible = s.series.filter((se) => !se.hidden);
    const hasRight = visible.some((se) => se.axis === "right");
    const rightLabel = hasRight && !s.compact && !!s.yLabelRight;
    const m = { l: s.compact ? 34 : 48, r: hasRight ? (rightLabel ? 62 : 48) : 12, t: 8, b: s.compact ? 20 : 30 };
    const iw = W - m.l - m.r;
    const ih = H - m.t - m.b;
    const logY = !!s.logY;

    // ---- domains (visible series only)
    let xmin = Infinity;
    let xmax = -Infinity;
    const yd = { left: [Infinity, -Infinity], right: [Infinity, -Infinity] };
    const inX = (xv: number) => !xr || (xv >= xr[0] && xv <= xr[1]);
    const upd = (ax: ChartAxis, v: number | null | undefined) => {
      if (v === null || v === undefined || !Number.isFinite(v) || (logY && v <= 0)) return;
      const d = yd[ax];
      if (v < d[0]) d[0] = v;
      if (v > d[1]) d[1] = v;
    };
    for (const se of visible) {
      const ax: ChartAxis = se.axis === "right" ? "right" : "left";
      const lines: { x: number[]; y: (number | null)[] }[] = se.y && se.y.length ? [{ x: se.x, y: se.y }] : [];
      for (const r of se.runs ?? []) lines.push(r);
      for (const d of lines) {
        for (let i = 0; i < d.x.length; i++) {
          const xv = d.x[i];
          if (!inX(xv)) continue;
          if (xv < xmin) xmin = xv;
          if (xv > xmax) xmax = xv;
          upd(ax, d.y[i]);
        }
      }
      if (se.lo && se.hi) {
        for (let i = 0; i < se.x.length; i++) {
          if (!inX(se.x[i])) continue;
          upd(ax, se.lo[i]);
          upd(ax, se.hi[i]);
        }
      }
    }
    if (xr) {
      xmin = xr[0];
      xmax = xr[1];
    }
    if (!Number.isFinite(xmin)) {
      const t = el("text", { x: W / 2, y: H / 2, "text-anchor": "middle", fill: theme.text, style: `font:${theme.font}`, class: "mc-empty" }, svg);
      t.textContent = s.emptyText || "No data";
      return;
    }
    const tf = logY ? (v: number) => Math.log10(Math.max(v, 1e-12)) : (v: number) => v;
    const scales = {} as Record<ChartAxis, { a: number; b: number }>;
    for (const ax of ["left", "right"] as const) {
      let [a, b] = yd[ax];
      if (!Number.isFinite(a)) {
        a = logY ? 1 : 0;
        b = logY ? 10 : 1;
      }
      a = tf(a);
      b = tf(b);
      const pad = (b - a || 1) * 0.06;
      scales[ax] = { a: a - pad, b: b + pad };
    }
    const xspan = xmax - xmin || 1;
    const X = (v: number) => m.l + ((v - xmin) / xspan) * iw;
    const Ys = {
      left: (v: number) => m.t + ih - ((tf(v) - scales.left.a) / (scales.left.b - scales.left.a)) * ih,
      right: (v: number) => m.t + ih - ((tf(v) - scales.right.a) / (scales.right.b - scales.right.a)) * ih,
    };
    geom = { W, H, m, iw, ih, xmin, xmax, invX: (px) => xmin + ((px - m.l) / iw) * (xmax - xmin) };

    // ---- grid and axes
    const defs = el("defs", {}, svg);
    const clip = el("clipPath", { id: clipId }, defs);
    el("rect", { x: m.l, y: m.t, width: iw, height: ih }, clip);
    const g = el("g", { class: "mc-axes" }, svg);
    const textStyle = `font:${theme.font}`;
    const xFormat = s.xFormat ?? fmtTick;
    for (const v of niceTicks(xmin, xmax, Math.max(2, Math.floor(iw / 90)))) {
      el("line", { x1: X(v), x2: X(v), y1: m.t, y2: m.t + ih, stroke: theme.grid }, g);
      const t = el("text", { x: X(v), y: m.t + ih + 14, "text-anchor": "middle", fill: theme.text, style: textStyle, class: "mc-xtick" }, g);
      t.textContent = xFormat(v);
    }
    const yticks = (ax: ChartAxis): number[] => {
      const sc = scales[ax];
      if (!logY) return niceTicks(sc.a, sc.b, Math.max(2, Math.floor(ih / 40)));
      const out: number[] = [];
      for (let p = Math.ceil(sc.a); p <= sc.b; p++) out.push(Math.pow(10, p));
      if (out.length >= 2) return out;
      // Less than one decade: linear ticks inside the range.
      return niceTicks(Math.pow(10, sc.a), Math.pow(10, sc.b), Math.max(2, Math.floor(ih / 40))).filter((v) => v > 0);
    };
    for (const v of yticks("left")) {
      el("line", { x1: m.l, x2: m.l + iw, y1: Ys.left(v), y2: Ys.left(v), stroke: theme.grid }, g);
      const t = el("text", { x: m.l - 6, y: Ys.left(v) + 3.5, "text-anchor": "end", fill: theme.text, style: textStyle, class: "mc-ytick-left" }, g);
      t.textContent = fmtTick(v);
    }
    if (hasRight) {
      for (const v of yticks("right")) {
        const t = el("text", { x: m.l + iw + 6, y: Ys.right(v) + 3.5, "text-anchor": "start", fill: theme.text, style: textStyle, class: "mc-ytick-right" }, g);
        t.textContent = fmtTick(v);
      }
      el("line", { x1: m.l + iw, x2: m.l + iw, y1: m.t, y2: m.t + ih, stroke: theme.axis, class: "mc-axis-right" }, g);
    }
    el("line", { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih, stroke: theme.axis, class: "mc-axis-x" }, g);
    if (!s.compact && s.yLabel) {
      const cy = m.t + ih / 2;
      const t = el("text", { x: 11, y: cy, transform: `rotate(-90 11 ${cy})`, "text-anchor": "middle", fill: theme.text, style: `${textStyle};opacity:.8`, class: "mc-ylabel-left" }, g);
      t.textContent = s.yLabel;
    }
    if (rightLabel) {
      const cx = W - 10;
      const cy = m.t + ih / 2;
      const t = el("text", { x: cx, y: cy, transform: `rotate(90 ${cx} ${cy})`, "text-anchor": "middle", fill: theme.text, style: `${textStyle};opacity:.8`, class: "mc-ylabel-right" }, g);
      t.textContent = s.yLabelRight!;
    }

    // ---- data (one <path> per line)
    const plot = el("g", { "clip-path": `url(#${clipId})`, class: "mc-data" }, svg);
    for (const se of visible) {
      const Y = se.axis === "right" ? Ys.right : Ys.left;
      const hasCenter = !!(se.y && se.y.length);
      for (const r of se.runs ?? []) {
        el(
          "path",
          {
            d: buildPath(r.x, r.y, X, Y, logY),
            fill: "none",
            stroke: se.color,
            "stroke-width": se.runWidth ?? 1,
            "stroke-opacity": se.runOpacity ?? (hasCenter ? 0.28 : 0.85),
            "stroke-dasharray": se.dash ?? "",
            class: "mc-run",
          },
          plot,
        );
      }
      if (se.lo && se.hi) {
        const d = buildBandPath(se.x, se.lo, se.hi, X, Y, logY);
        if (d) el("path", { d, fill: se.color, "fill-opacity": se.bandOpacity ?? 0.16, stroke: "none", class: "mc-band" }, plot);
      }
      if (hasCenter) {
        el(
          "path",
          {
            d: buildPath(se.x, se.y!, X, Y, logY),
            fill: "none",
            stroke: se.color,
            "stroke-width": se.width ?? 2,
            "stroke-dasharray": se.dash ?? "",
            "stroke-linejoin": "round",
            class: "mc-center",
          },
          plot,
        );
      }
    }

    // ---- hover layer (updated on mouse move without touching the data)
    const hl = el("g", { class: "mc-hover", "data-export": "skip" }, svg);
    hover = {
      cross: el("line", { y1: m.t, y2: m.t + ih, stroke: theme.axis, "stroke-dasharray": "3 3", visibility: "hidden" }, hl),
      sel: el("rect", { y: m.t, height: ih, fill: theme.axis, "fill-opacity": 0.12, visibility: "hidden" }, hl),
    };
    el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent", style: "cursor:crosshair", class: "mc-hit", "data-export": "skip" }, svg);
    if (xr) {
      const b = el("text", { x: m.l + iw - 4, y: m.t + 12, "text-anchor": "end", fill: theme.text, style: `${textStyle};opacity:.7;cursor:pointer`, class: "mc-reset", "data-export": "skip" }, svg);
      b.textContent = "⟲ reset zoom";
      b.addEventListener("click", (ev) => {
        ev.stopPropagation();
        resetZoom();
      });
    }
  }

  // ---------------------------------------------------------------- interactions

  const localX = (ev: MouseEvent) => ev.clientX - svg.getBoundingClientRect().left;
  const inPlot = (ev: MouseEvent) => {
    if (!geom) return false;
    const rect = svg.getBoundingClientRect();
    const px = ev.clientX - rect.left;
    const py = ev.clientY - rect.top;
    return px >= geom.m.l && px <= geom.m.l + geom.iw && py >= geom.m.t && py <= geom.m.t + geom.ih;
  };

  function hideHover(): void {
    hover?.cross.setAttribute("visibility", "hidden");
    hover?.sel.setAttribute("visibility", "hidden");
    tip.style.display = "none";
  }

  /** @ai-generated */
  function onMove(ev: MouseEvent): void {
    if (!geom || !hover) return;
    if (!inPlot(ev) && dragStart === null) {
      hideHover();
      return;
    }
    const px = Math.max(geom.m.l, Math.min(geom.m.l + geom.iw, localX(ev)));
    const xv = geom.invX(px);
    hover.cross.setAttribute("x1", String(px));
    hover.cross.setAttribute("x2", String(px));
    hover.cross.setAttribute("visibility", "visible");
    if (dragStart !== null) {
      hover.sel.setAttribute("x", String(Math.min(dragStart, px)));
      hover.sel.setAttribute("width", String(Math.abs(px - dragStart)));
      hover.sel.setAttribute("visibility", "visible");
    }
    const rows: string[] = [];
    let xshown: number | null = null;
    let best = Infinity;
    const consider = (x: number) => {
      const d = Math.abs(x - xv);
      if (d < best) {
        best = d;
        xshown = x;
      }
    };
    const swatch = (c: string) => `<span style="display:inline-block;width:10px;height:3px;background:${escapeHtml(c)};vertical-align:middle;margin-right:6px"></span>`;
    for (const se of spec.series) {
      if (se.hidden) continue;
      if (se.y && se.y.length) {
        const i = nearestIndex(se.x, xv);
        if (i < 0) continue;
        consider(se.x[i]);
        const band = se.lo && se.hi && se.lo[i] !== null && se.lo[i] !== undefined ? ` <span style="opacity:.6">[${fmtTick(se.lo[i]!)}, ${fmtTick(se.hi[i] ?? NaN)}]</span>` : "";
        rows.push(`<div>${swatch(se.color)}${escapeHtml(se.label)}: <b>${fmtVal(se.y[i])}</b>${band}</div>`);
      } else {
        for (const r of se.runs ?? []) {
          const i = nearestIndex(r.x, xv);
          if (i < 0) continue;
          consider(r.x[i]);
          const who = r.label ?? `seed ${r.seed ?? "?"}`;
          rows.push(`<div>${swatch(se.color)}${escapeHtml(se.label)} · ${escapeHtml(who)}: <b>${fmtVal(r.y[i])}</b></div>`);
        }
      }
    }
    const xName = spec.xLabel ?? "step";
    const xs = xshown === null ? "" : (xshown as number).toLocaleString("en-US");
    tip.innerHTML =
      `<div style="opacity:.7;margin-bottom:2px">${escapeHtml(xName)} ${xs}</div>` +
      rows.slice(0, 12).join("") +
      (rows.length > 12 ? `<div style="opacity:.6">+${rows.length - 12} more</div>` : "") +
      (spec.onPointClick ? `<div style="opacity:.55;margin-top:3px">click → episodes at this step</div>` : "");
    tip.style.display = "block";
    const tw = tip.offsetWidth;
    const rect = svg.getBoundingClientRect();
    tip.style.left = `${px + 14 + tw > geom.W ? px - tw - 14 : px + 14}px`;
    tip.style.top = `${Math.max(0, ev.clientY - rect.top - 20)}px`;
  }

  function onDown(ev: MouseEvent): void {
    if (ev.button !== 0 || !inPlot(ev)) return;
    dragStart = localX(ev);
  }

  /** @ai-generated */
  function onUp(ev: MouseEvent): void {
    if (!geom || dragStart === null) return;
    const px = Math.max(geom.m.l, Math.min(geom.m.l + geom.iw, localX(ev)));
    const start = dragStart;
    dragStart = null;
    hover?.sel.setAttribute("visibility", "hidden");
    if (Math.abs(px - start) > 8) {
      xr = [geom.invX(Math.min(px, start)), geom.invX(Math.max(px, start))];
      draw();
      return;
    }
    if (spec.onPointClick) {
      const idx = spec.series.findIndex((se) => !se.hidden && se.x.length);
      if (idx >= 0) {
        const se = spec.series[idx];
        const i = nearestIndex(se.x, geom.invX(px));
        spec.onPointClick(se.x[i], idx);
      }
    }
  }

  function onLeave(): void {
    dragStart = null;
    hideHover();
  }

  function onDbl(ev: MouseEvent): void {
    if (inPlot(ev)) resetZoom();
  }

  function resetZoom(): void {
    if (!xr) return;
    xr = null;
    draw();
  }

  svg.addEventListener("mousemove", onMove);
  svg.addEventListener("mousedown", onDown);
  svg.addEventListener("mouseup", onUp);
  svg.addEventListener("mouseleave", onLeave);
  svg.addEventListener("dblclick", onDbl);

  /** @ai-generated */
  function exportSVG(): string {
    const { W, H } = size();
    const clone = svg.cloneNode(true) as SVGSVGElement;
    clone.querySelectorAll('[data-export="skip"]').forEach((n) => n.remove());
    clone.setAttribute("xmlns", NS);
    clone.setAttribute("width", String(W));
    clone.setAttribute("height", String(H));
    clone.setAttribute("viewBox", `0 0 ${W} ${H}`);
    clone.removeAttribute("style");
    const bg = document.createElementNS(NS, "rect");
    bg.setAttribute("width", "100%");
    bg.setAttribute("height", "100%");
    bg.setAttribute("fill", theme.bg);
    clone.insertBefore(bg, clone.firstChild);
    return new XMLSerializer().serializeToString(clone);
  }

  /** @ai-generated */
  function exportPNG(scale = 2): Promise<Blob> {
    const { W, H } = size();
    const url = URL.createObjectURL(new Blob([exportSVG()], { type: "image/svg+xml;charset=utf-8" }));
    return new Promise<Blob>((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = Math.round(W * scale);
        canvas.height = Math.round(H * scale);
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          URL.revokeObjectURL(url);
          reject(new Error("Canvas 2D context unavailable"));
          return;
        }
        ctx.scale(scale, scale);
        ctx.drawImage(img, 0, 0, W, H);
        URL.revokeObjectURL(url);
        canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("PNG encoding failed"))), "image/png");
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error("Could not rasterise the SVG"));
      };
      img.src = url;
    });
  }

  draw();
  return {
    update(next: ChartSpec) {
      spec = next;
      draw();
    },
    redraw: draw,
    resetZoom,
    get zoom() {
      return xr;
    },
    exportSVG,
    exportPNG,
    destroy() {
      svg.remove();
      tip.remove();
    },
  };
}

// ---------------------------------------------------------------- sparkline

export type SparklineOptions = { w?: number; h?: number; lo?: (number | null)[] | null; hi?: (number | null)[] | null; fill?: boolean };

/**
 * Paths of a sparkline (line, optional band, optional area fill) in a `w × h` box.
 *
 * @ai-generated
 */
export function sparklinePaths(xs: number[], ys: (number | null)[], opts: SparklineOptions = {}): { line: string; band: string; area: string } {
  const w = opts.w ?? 120;
  const h = opts.h ?? 28;
  const vals = ys.filter((v): v is number => v !== null && Number.isFinite(v));
  if (!vals.length || !xs.length) return { line: "", band: "", area: "" };
  const all = [...vals];
  for (const arr of [opts.lo, opts.hi]) for (const v of arr ?? []) if (v !== null && Number.isFinite(v)) all.push(v);
  let a = Math.min(...all);
  let b = Math.max(...all);
  if (a === b) {
    a -= 1;
    b += 1;
  }
  const x0 = xs[0];
  const x1 = xs[xs.length - 1];
  const X = (v: number) => ((v - x0) / (x1 - x0 || 1)) * (w - 2) + 1;
  const Y = (v: number) => h - 2 - ((v - a) / (b - a)) * (h - 4);
  const line = buildPath(xs, ys, X, Y);
  const band = opts.lo && opts.hi ? buildBandPath(xs, opts.lo, opts.hi, X, Y) : "";
  let area = "";
  if ((opts.fill ?? true) && !band && line) {
    const firstX = xs[ys.findIndex((v) => v !== null && Number.isFinite(v))];
    let lastI = ys.length - 1;
    while (lastI > 0 && (ys[lastI] === null || !Number.isFinite(ys[lastI]!))) lastI--;
    area = `${line}L${r1(X(xs[lastI]))},${h}L${r1(X(firstX))},${h}Z`;
  }
  return { line, band, area };
}
