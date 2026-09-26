/**
 * Series micro-batcher: every `fetch(query)` issued during the same tick is sent as a single
 * `POST /api/series` (chunked by `maxBatch`). Each caller gets its own promise; a failed query
 * resolves to `{ok: false, issue}` without affecting the others. Only a failure of the whole HTTP
 * request rejects (every pending promise of that batch, with the same error).
 */
import { clientIssue, formatZodError, SeriesOutcomeSchema, type SeriesOutcome, type SeriesQuery } from "./schemas";

export const SERIES_DEFAULTS = {
  x: "time_step",
  runs: null,
  center: "mean",
  band: "ci95",
  resolution: null,
  include_runs: false,
  max_points: 1000,
} as const satisfies Omit<Required<SeriesQuery>, "experiment" | "table" | "metric">;

/** Fill a query with the contract defaults (runs sorted) so equal queries compare equal. @ai-generated */
export function normaliseQuery(q: SeriesQuery): Required<SeriesQuery> {
  const center = q.center ?? SERIES_DEFAULTS.center;
  return {
    experiment: q.experiment,
    table: q.table,
    metric: q.metric,
    x: q.x ?? SERIES_DEFAULTS.x,
    runs: q.runs ? [...q.runs].sort() : null,
    center,
    band: center === "none" ? "none" : (q.band ?? SERIES_DEFAULTS.band),
    resolution: q.resolution ?? null,
    include_runs: center === "none" ? true : (q.include_runs ?? false),
    max_points: q.max_points ?? SERIES_DEFAULTS.max_points,
  };
}

/** Canonical JSON key of a query (defaults filled, keys in a fixed order). @ai-generated */
export function queryKey(q: SeriesQuery): string {
  const n = normaliseQuery(q);
  return JSON.stringify([n.experiment, n.table, n.metric, n.x, n.runs, n.center, n.band, n.resolution, n.include_runs, n.max_points]);
}

export type SendBatch = (queries: SeriesQuery[], signal?: AbortSignal) => Promise<unknown[]>;

type Pending = {
  query: SeriesQuery;
  key: string;
  resolve: (o: SeriesOutcome) => void;
  reject: (e: unknown) => void;
  signal?: AbortSignal;
};

export type BatcherOptions = {
  /** Maximum queries per request (contract: 1..200). */
  maxBatch?: number;
  /** Schedules the flush; defaults to `setTimeout(fn, 0)` so the whole tick is collected. */
  schedule?: (fn: () => void) => void;
};

/**
 * Parse one item of a batch response; malformed items become `{ok: false}` client issues.
 *
 * @ai-generated
 */
export function parseOutcome(raw: unknown): SeriesOutcome {
  const r = SeriesOutcomeSchema.safeParse(raw);
  if (r.success) return r.data as SeriesOutcome;
  return { ok: false, issue: clientIssue("malformed-series", "The series result could not be read", formatZodError(r.error)) };
}

export class SeriesBatcher {
  private queue: Pending[] = [];
  private scheduled = false;
  private readonly maxBatch: number;
  private readonly schedule: (fn: () => void) => void;
  /** Number of HTTP requests sent (for tests and diagnostics). */
  requestCount = 0;

  constructor(
    private readonly send: SendBatch,
    opts: BatcherOptions = {},
  ) {
    this.maxBatch = Math.max(1, Math.min(200, opts.maxBatch ?? 200));
    this.schedule = opts.schedule ?? ((fn) => setTimeout(fn, 0));
  }

  /**
   * Queue a query for the next flush. The promise resolves with the outcome of this query, or
   * rejects with an `AbortError` when `signal` aborts before the response arrives.
   *
   * @ai-generated
   */
  fetch(query: SeriesQuery, signal?: AbortSignal): Promise<SeriesOutcome> {
    if (signal?.aborted) return Promise.reject(abortError());
    return new Promise<SeriesOutcome>((resolve, reject) => {
      const p: Pending = { query, key: queryKey(query), resolve, reject, signal };
      signal?.addEventListener(
        "abort",
        () => {
          this.queue = this.queue.filter((x) => x !== p);
          reject(abortError());
        },
        { once: true },
      );
      this.queue.push(p);
      if (!this.scheduled) {
        this.scheduled = true;
        this.schedule(() => this.flush());
      }
    });
  }

  /** Send everything queued now: identical queries are deduplicated, then chunked. @ai-generated */
  flush(): void {
    this.scheduled = false;
    const pending = this.queue.filter((p) => !p.signal?.aborted);
    this.queue = [];
    if (!pending.length) return;
    const groups = new Map<string, Pending[]>();
    for (const p of pending) {
      const g = groups.get(p.key);
      if (g) g.push(p);
      else groups.set(p.key, [p]);
    }
    const unique = [...groups.values()];
    for (let i = 0; i < unique.length; i += this.maxBatch) {
      void this.sendChunk(unique.slice(i, i + this.maxBatch));
    }
  }

  /** @ai-generated */
  private async sendChunk(chunk: Pending[][]): Promise<void> {
    this.requestCount++;
    let items: unknown[];
    try {
      items = await this.send(chunk.map((g) => g[0].query));
    } catch (e) {
      chunk.flat().forEach((p) => p.reject(e));
      return;
    }
    chunk.forEach((group, i) => {
      const outcome: SeriesOutcome =
        Array.isArray(items) && i < items.length
          ? parseOutcome(items[i])
          : { ok: false, issue: clientIssue("missing-series", "The server returned no result for this query") };
      group.forEach((p) => p.resolve(outcome));
    });
  }
}

function abortError(): DOMException {
  return new DOMException("The operation was aborted.", "AbortError");
}
