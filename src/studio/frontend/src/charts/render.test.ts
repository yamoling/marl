import { describe, expect, it, vi } from "vitest";
import { buildPath, mount, nearestIndex, niceTicks, sparklinePaths, type ChartSeriesInput, type ChartSpec } from "./render";

const W = 600;
const H = 300;

function render(spec: ChartSpec) {
  const div = document.createElement("div");
  document.body.appendChild(div);
  const h = mount(div, spec, { width: W, height: H });
  const q = (sel: string) => div.querySelectorAll(sel);
  return { div, h, q };
}

const series = (over: Partial<ChartSeriesInput> = {}): ChartSeriesInput => ({
  label: "s",
  color: "#4e79a7",
  x: [0, 10, 20, 30],
  y: [1, 2, 3, 4],
  lo: [0.5, 1.5, 2.5, 3.5],
  hi: [1.5, 2.5, 3.5, 4.5],
  runs: [
    { seed: 0, x: [0, 10, 20, 30], y: [1, 2, 3, 4] },
    { seed: 1, x: [0, 10, 20, 30], y: [1, 2, null, 4] },
  ],
  ...over,
});

describe("helpers", () => {
  it("niceTicks and nearestIndex", () => {
    expect(niceTicks(0, 1, 5)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
    const xs = [0, 10, 20, 40];
    expect([-5, 4, 6, 29, 31, 100].map((v) => nearestIndex(xs, v))).toEqual([0, 0, 1, 2, 3, 3]);
    expect(nearestIndex([], 1)).toBe(-1);
  });
  it("buildPath lifts the pen on null and on ≤ 0 in log scale", () => {
    const id = (v: number) => v;
    expect(buildPath([0, 1, 2, 3], [1, null, 2, 3], id, id)).toBe("M0,1M2,2L3,3");
    expect(buildPath([0, 1, 2], [-1, 1, 2], id, id, true)).toBe("M1,1L2,2");
  });
  it("sparkline paths", () => {
    const p = sparklinePaths([0, 1, 2], [1, 2, 3], { w: 100, h: 20 });
    expect(p.line.startsWith("M1,")).toBe(true);
    expect(p.area.endsWith("Z")).toBe(true);
    expect(sparklinePaths([], [])).toEqual({ line: "", band: "", area: "" });
  });
});

describe("render structure", () => {
  it("draws one path per run, a band and a centre line; no right axis by default", () => {
    const { q } = render({ series: [series(), series({ label: "t", runs: [] })], yLabel: "score" });
    expect(q("path.mc-run")).toHaveLength(2);
    expect(q("path.mc-band")).toHaveLength(2);
    expect(q("path.mc-center")).toHaveLength(2);
    expect(q(".mc-ytick-left").length).toBeGreaterThan(1);
    expect(q(".mc-xtick").length).toBeGreaterThan(1);
    expect(q(".mc-ytick-right")).toHaveLength(0);
    expect(q(".mc-ylabel-left")[0].textContent).toBe("score");
    expect(q(".mc-ylabel-right")).toHaveLength(0);
  });

  it("the right axis (ticks and label) appears only when a visible series uses it", () => {
    const right = series({ axis: "right", y: [100, 200, 300, 400], lo: null, hi: null, runs: [] });
    const r1 = render({ series: [series(), right], yLabelRight: "epsilon" });
    expect(r1.q(".mc-ytick-right").length).toBeGreaterThan(1);
    expect(r1.q(".mc-ylabel-right")[0].textContent).toBe("epsilon");
    const r2 = render({ series: [series(), { ...right, hidden: true }], yLabelRight: "epsilon" });
    expect(r2.q(".mc-ytick-right")).toHaveLength(0);
  });

  it("hidden series are neither drawn nor part of the y domain", () => {
    const big = series({ label: "big", y: [1000, 2000, 3000, 4000], lo: null, hi: null, runs: [], hidden: true });
    const { q } = render({ series: [series({ runs: [] }), big] });
    expect(q("path.mc-center")).toHaveLength(1);
    const ticks = [...q(".mc-ytick-left")].map((t) => t.textContent);
    expect(ticks.some((t) => t!.endsWith("k"))).toBe(false);
  });

  it("log y ignores values ≤ 0 and uses decade ticks", () => {
    const s = series({ y: [-1, 0, 10, 1000], lo: null, hi: null, runs: [] });
    const { q } = render({ series: [s], logY: true });
    const d = q("path.mc-center")[0].getAttribute("d")!;
    expect(d.match(/[ML]/g)).toHaveLength(2);
    const ticks = [...q(".mc-ytick-left")].map((t) => t.textContent);
    expect(ticks).toEqual(["10", "100", "1k"]);
  });

  it("empty text, update, export and destroy", () => {
    const { div, h, q } = render({ series: [], emptyText: "Drop a metric on the Y shelf" });
    expect(q(".mc-empty")[0].textContent).toBe("Drop a metric on the Y shelf");
    h.update({ series: [series()] });
    expect(q(".mc-empty")).toHaveLength(0);
    const svg = h.exportSVG();
    expect(svg).toContain('xmlns="http://www.w3.org/2000/svg"');
    expect(svg).toContain("mc-center");
    expect(svg).not.toContain("mc-hit");
    h.destroy();
    expect(div.querySelector("svg")).toBeNull();
  });

  it("point click reports the nearest x; drag zooms and dblclick resets", () => {
    const onPointClick = vi.fn();
    const { div, h } = render({ series: [series()], onPointClick });
    const svg = div.querySelector("svg")!;
    svg.getBoundingClientRect = () => ({ left: 0, top: 0, right: W, bottom: H, width: W, height: H, x: 0, y: 0, toJSON: () => ({}) });
    const fire = (type: string, x: number) => svg.dispatchEvent(new MouseEvent(type, { clientX: x, clientY: 100, bubbles: true }));
    fire("mousedown", 540);
    fire("mouseup", 540);
    expect(onPointClick).toHaveBeenCalledWith(30, 0);
    fire("mousemove", 300);
    expect(div.querySelector<HTMLElement>(".mc-tip")!.style.display).toBe("block");
    expect(div.querySelector(".mc-tip")!.textContent).toContain("step");
    fire("mousedown", 100);
    fire("mouseup", 300);
    expect(h.zoom).not.toBeNull();
    fire("dblclick", 200);
    expect(h.zoom).toBeNull();
  });
});

describe("performance budget", () => {
  // 8 experiments × 2 metrics × 10 runs = 160 run paths of 1000 points (+ 16 centre lines and bands).
  const N = 1000;
  const xs = Array.from({ length: N }, (_, i) => i * 1000);
  const line = (k: number) => xs.map((_, i) => Math.sin(i / 50 + k) + k * 0.01);
  const big: ChartSeriesInput[] = Array.from({ length: 16 }, (_, s) => ({
    label: `s${s}`,
    color: "#4e79a7",
    axis: s % 2 ? "right" : "left",
    x: xs,
    y: line(s),
    lo: line(s).map((v) => v - 0.1),
    hi: line(s).map((v) => v + 0.1),
    runs: Array.from({ length: 10 }, (_, r) => ({ seed: r, x: xs, y: line(s * 10 + r) })),
  }));

  it("builds 160 × 1000-point paths in < 50 ms", () => {
    const X = (v: number) => v / 1000;
    const Y = (v: number) => v * 100;
    for (let i = 0; i < 3; i++) for (const s of big) for (const r of s.runs!) buildPath(r.x, r.y, X, Y); // warm-up
    const t0 = performance.now();
    let len = 0;
    for (const s of big) for (const r of s.runs!) len += buildPath(r.x, r.y, X, Y).length;
    const ms = performance.now() - t0;
    console.info(`[perf] path building, 160 × ${N} points: ${ms.toFixed(1)} ms`);
    expect(len).toBeGreaterThan(0);
    expect(ms).toBeLessThan(50);
  });

  it("full render (domains + DOM) is measured", () => {
    const div = document.createElement("div");
    mount(div, { series: big }, { width: W, height: H }); // warm-up
    const t0 = performance.now();
    mount(div, { series: big }, { width: W, height: H });
    const ms = performance.now() - t0;
    console.info(`[perf] full render in jsdom, 160 runs + 16 centres/bands: ${ms.toFixed(1)} ms`);
    expect(div.querySelectorAll("path.mc-run").length).toBeGreaterThanOrEqual(160);
    // jsdom's DOM is several times slower than a browser's; only guard against pathological cases here.
    expect(ms).toBeLessThan(500);
  });
});
