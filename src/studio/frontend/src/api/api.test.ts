import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, encodeId, request } from "./client";
import { backoffDelay, connectEvents, type EventSourceLike } from "./events";
import {
  CatalogSchema,
  ExperimentDetailSchema,
  ExperimentSummarySchema,
  IssueSchema,
  parseArray,
  RunStatusSchema,
  SeriesResultSchema,
} from "./schemas";
import { queryKey, SeriesBatcher } from "./series";

const summary = (id: string, extra: object = {}) => ({
  id,
  name: id,
  algo: "VDN",
  env: null,
  created: null,
  n_steps: 100,
  status: "COMPLETED",
  progress: 1,
  health: "ok",
  issue_counts: { info: 0, warning: 0, error: 0 },
  n_runs: 1,
  running_runs: 0,
  ...extra,
});

describe("schemas", () => {
  it("parseArray keeps good items and reports bad ones with index and raw", () => {
    const items = [summary("a"), { name: "no id" }, summary("b", { status: "EXPLODED", extra_key: 1 }), null];
    const { ok, bad } = parseArray(ExperimentSummarySchema, items);
    expect(ok.map((e) => e.id)).toEqual(["a", "b"]);
    expect(ok[1].status).toBe("UNKNOWN");
    expect((ok[1] as Record<string, unknown>).extra_key).toBe(1);
    expect(bad.map((b) => b.index)).toEqual([1, 3]);
    expect(bad[0].raw).toEqual({ name: "no id" });
    expect(bad[0].error).toMatch(/id/);
    expect(parseArray(ExperimentSummarySchema, "nope").bad).toHaveLength(1);
  });

  it("tolerant fields: catch defaults and derived names", () => {
    const s = ExperimentSummarySchema.parse({ id: "sweeps/x/exp-3", health: 12 });
    expect(s).toMatchObject({ name: "exp-3", health: "warning", n_runs: 0, issue_counts: { info: 0, warning: 0, error: 0 }, algo: null });
    expect(IssueSchema.parse({ level: "fatal" })).toMatchObject({ level: "warning", code: "unknown", path: null });
    expect(RunStatusSchema.parse("QUEUED")).toBe("UNKNOWN");
  });

  it("experiment detail: a malformed run becomes a placeholder with a client issue", () => {
    const d = ExperimentDetailSchema.parse({
      ...summary("e"),
      raw: "broken",
      issues: [{ level: "error", code: "x", message: "m", scope: "experiment" }, 5],
      capabilities: { metrics: true, params: "weird", replay: null, launch: false },
      runs: [
        { id: "e/run-0", dirname: "run-0", seed: 0, status: "RUNNING" },
        { dirname: "run-1", seed: "one" },
      ],
      params: [{ path: "a", kind: "number", value: 1, depth: 0 }, { kind: "number" }],
    });
    expect(d.raw).toEqual({});
    expect(d.capabilities).toMatchObject({ params: "none", launch: false });
    expect(d.runs.map((r) => [r.id, r.status])).toEqual([
      ["e/run-0", "RUNNING"],
      ["e/run-1", "UNKNOWN"],
    ]);
    expect(d.runs[1].issues[0].code).toBe("malformed-run");
    expect(d.params.map((p) => p.path)).toEqual(["a"]);
    expect(d.issues.map((i) => i.code)).toEqual(["x", "malformed-issue", "malformed-run", "malformed-params"]);
  });

  it("catalog drops malformed tables; series results require x", () => {
    const c = CatalogSchema.parse({ tables: { test: { metrics: ["score"] }, bad: 3 }, default_metric: "nope" });
    expect(Object.keys(c.tables)).toEqual(["test"]);
    expect(c.default_metric).toBeNull();
    expect(SeriesResultSchema.safeParse({ center: [] }).success).toBe(false);
    const r = SeriesResultSchema.parse({ x: [0, 1], center: [null, 2], runs: [{ run: "r", x: [0], y: [1] }, { bad: 1 }] });
    expect(r.runs).toHaveLength(1);
    expect(r.missing_runs).toEqual([]);
  });
});

describe("client", () => {
  it("encodes ids per segment", () => {
    expect(encodeId("sweeps/a b/exp#3")).toBe("sweeps/a%20b/exp%233");
  });

  it("maps ErrorBody to ApiError and validates responses", async () => {
    const f = vi.fn(async (url: string) => {
      if (url.includes("bad"))
        return new Response(JSON.stringify({ error: "seed-collision", message: "Seeds 3, 4 exist" }), { status: 409 });
      if (url.includes("none")) return new Response(null, { status: 204 });
      return new Response(JSON.stringify([1, "x", 2]), { status: 200 });
    });
    const err = await request("GET", "/bad", { fetchImpl: f as unknown as typeof fetch }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ status: 409, code: "seed-collision", message: "Seeds 3, 4 exist" });
    expect(await request("POST", "/none", { fetchImpl: f as unknown as typeof fetch })).toBeUndefined();
    const { TestStepsSchema } = await import("./schemas");
    expect(await request("GET", "/steps", { schema: TestStepsSchema, fetchImpl: f as unknown as typeof fetch })).toEqual([1, 2]);
    const net = await request("GET", "/x", { fetchImpl: (() => Promise.reject(new TypeError("offline"))) as typeof fetch }).catch((e) => e);
    expect(net).toMatchObject({ status: 0, code: "network" });
  });
});

describe("series micro-batcher", () => {
  const q = (metric: string, experiment = "e") => ({ experiment, table: "test", metric });
  const res = { x: [0], center: [1], lo: null, hi: null, n: [1], runs: [], used_runs: [], missing_runs: [], resolution: 1, issues: [] };

  it("queries of the same tick share one request; results resolve per query; duplicates are merged", async () => {
    const send = vi.fn(async (qs: { metric: string }[]) =>
      qs.map((x) =>
        x.metric === "bad"
          ? { ok: false, issue: { level: "error", code: "nope", message: "no", scope: "" } }
          : x.metric === "garbled"
            ? { ok: true }
            : { ok: true, result: res },
      ),
    );
    const b = new SeriesBatcher(send);
    const out = await Promise.all([b.fetch(q("a")), b.fetch(q("bad")), b.fetch(q("a")), b.fetch(q("garbled"))]);
    expect(send).toHaveBeenCalledTimes(1);
    expect(send.mock.calls[0][0]).toHaveLength(3);
    expect(out.map((o) => o.ok)).toEqual([true, false, true, false]);
    expect(out[3].ok === false && out[3].issue.code).toBe("malformed-series");
  });

  it("chunks by maxBatch, rejects on transport errors, and supports abort", async () => {
    const send = vi.fn(async (qs: unknown[]) => qs.map(() => ({ ok: true, result: res })));
    const b = new SeriesBatcher(send, { maxBatch: 2 });
    await Promise.all(["a", "b", "c"].map((m) => b.fetch(q(m))));
    expect(send).toHaveBeenCalledTimes(2);

    const failing = new SeriesBatcher(async () => {
      throw new ApiError(500, "http-500", "down");
    });
    await expect(failing.fetch(q("a"))).rejects.toMatchObject({ status: 500 });

    const ac = new AbortController();
    const p = b.fetch(q("z"), ac.signal);
    ac.abort();
    await expect(p).rejects.toMatchObject({ name: "AbortError" });
  });

  it("missing items in the response become client issues", async () => {
    const b = new SeriesBatcher(async () => []);
    const o = await b.fetch(q("a"));
    expect(o.ok).toBe(false);
  });

  it("queryKey normalises defaults", () => {
    expect(queryKey({ ...q("a"), center: "mean", band: "ci95", runs: null })).toBe(queryKey(q("a")));
    expect(queryKey({ ...q("a"), runs: ["r2", "r1"] })).toBe(queryKey({ ...q("a"), runs: ["r1", "r2"] }));
    expect(queryKey({ ...q("a"), center: "none" })).not.toBe(queryKey(q("a")));
  });
});

describe("events", () => {
  afterEach(() => vi.useRealTimers());

  class FakeSource implements EventSourceLike {
    static all: FakeSource[] = [];
    listeners = new Map<string, (ev: MessageEvent) => void>();
    onopen: ((ev: Event) => void) | null = null;
    onerror: ((ev: Event) => void) | null = null;
    closed = false;
    constructor() {
      FakeSource.all.push(this);
    }
    addEventListener(type: string, l: (ev: MessageEvent) => void) {
      this.listeners.set(type, l);
    }
    close() {
      this.closed = true;
    }
    emit(type: string, data: unknown) {
      this.listeners.get(type)?.({ data: typeof data === "string" ? data : JSON.stringify(data) } as MessageEvent);
    }
  }

  it("validates and dispatches late launch-failed events", () => {
    FakeSource.all = [];
    const failed = vi.fn();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const conn = connectEvents({ "launch-failed": failed }, { createSource: () => new FakeSource() });
    const source = FakeSource.all[0];
    const issue = { level: "error", code: "launch-failed", message: "Launcher exited", scope: "run:run-0", path: null, detail: "stderr" };
    source.emit("launch-failed", { experiment: "e", runs: ["e/run-0"], issue });
    source.emit("launch-failed", { experiment: "e", runs: "e/run-0", issue });
    source.emit("launch-failed", { experiment: "e", runs: ["e/run-0"] });
    expect(failed).toHaveBeenCalledOnce();
    expect(failed).toHaveBeenCalledWith({ experiment: "e", runs: ["e/run-0"], issue });
    expect(warn).toHaveBeenCalledTimes(2);
    conn.close();
    warn.mockRestore();
  });

  it("backoff doubles from 1 s and caps at 30 s", () => {
    expect([0, 1, 2, 3, 4, 5, 6].map((a) => backoffDelay(a))).toEqual([1000, 2000, 4000, 8000, 16000, 30000, 30000]);
  });

  it("dispatches typed events, skips bad ones, and reconnects with backoff", () => {
    vi.useFakeTimers();
    vi.spyOn(console, "warn").mockImplementation(() => {});
    FakeSource.all = [];
    const progress = vi.fn();
    const states: string[] = [];
    const conn = connectEvents({ "run-progress": progress, state: (s) => states.push(s) }, { createSource: () => new FakeSource() });
    const s1 = FakeSource.all[0];
    s1.onopen?.(new Event("open"));
    s1.emit("run-progress", { experiment: "e", run: "e/run-0", status: "RUNNING", progress: 0.5, latest_step: 10 });
    s1.emit("run-progress", { nope: true });
    s1.emit("run-progress", "{not json");
    expect(progress).toHaveBeenCalledTimes(1);
    expect(progress.mock.calls[0][0]).toMatchObject({ run: "e/run-0", progress: 0.5 });

    s1.onerror?.(new Event("error"));
    expect(s1.closed).toBe(true);
    expect(conn.state).toBe("paused");
    vi.advanceTimersByTime(999);
    expect(FakeSource.all).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(FakeSource.all).toHaveLength(2);
    FakeSource.all[1].onerror?.(new Event("error"));
    vi.advanceTimersByTime(1999);
    expect(FakeSource.all).toHaveLength(2);
    vi.advanceTimersByTime(1);
    expect(FakeSource.all).toHaveLength(3);
    conn.close();
    expect(FakeSource.all[2].closed).toBe(true);
    expect(states).toEqual(["open", "paused", "connecting", "paused", "connecting", "closed"]);
  });
});
