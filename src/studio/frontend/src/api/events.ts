/**
 * Server-Sent Events client for `GET /api/events` with exponential-backoff reconnection
 * (1 s → 30 s), typed handlers and per-message validation (bad messages are logged and skipped).
 */
import { EVENT_SCHEMAS, formatZodError, type LiveEventMap, type LiveEventName } from "./schemas";

export type ConnectionState = "connecting" | "open" | "paused" | "closed";

export type LiveHandlers = { [K in LiveEventName]?: (data: LiveEventMap[K]) => void } & {
  /** Connection state changes; `paused` means disconnected and waiting to retry. */
  state?: (state: ConnectionState) => void;
};

export interface LiveConnection {
  readonly state: ConnectionState;
  close(): void;
}

/** Minimal EventSource surface, so tests can inject a fake. */
export interface EventSourceLike {
  addEventListener(type: string, listener: (ev: MessageEvent) => void): void;
  onopen: ((ev: Event) => void) | null;
  onerror: ((ev: Event) => void) | null;
  close(): void;
}

export type EventsOptions = {
  url?: string;
  createSource?: (url: string) => EventSourceLike;
  minDelayMs?: number;
  maxDelayMs?: number;
};

/** Delay before reconnection attempt `attempt` (0-based): min·2^attempt capped at max. @ai-generated */
export function backoffDelay(attempt: number, minMs = 1000, maxMs = 30000): number {
  return Math.min(maxMs, minMs * 2 ** Math.max(0, attempt));
}

/**
 * Dispatch one raw SSE message to its typed handler after validation.
 *
 * @ai-generated
 */
export function dispatchEvent(name: LiveEventName, raw: string, handlers: LiveHandlers): void {
  let json: unknown;
  try {
    json = JSON.parse(raw || "{}");
  } catch {
    console.warn(`[events] ${name}: invalid JSON`, raw);
    return;
  }
  const parsed = EVENT_SCHEMAS[name].safeParse(json);
  if (!parsed.success) {
    console.warn(`[events] ${name}: ${formatZodError(parsed.error)}`, json);
    return;
  }
  const handler = handlers[name] as ((d: unknown) => void) | undefined;
  handler?.(parsed.data);
}

/**
 * Open the live event stream. The connection reconnects by itself until `close()` is called.
 *
 * @ai-generated
 */
export function connectEvents(handlers: LiveHandlers, opts: EventsOptions = {}): LiveConnection {
  const url = opts.url ?? "/api/events";
  const create = opts.createSource ?? ((u: string) => new EventSource(u) as unknown as EventSourceLike);
  const minMs = opts.minDelayMs ?? 1000;
  const maxMs = opts.maxDelayMs ?? 30000;
  let source: EventSourceLike | null = null;
  let attempt = 0;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let state: ConnectionState = "connecting";

  const setState = (s: ConnectionState) => {
    if (s === state) return;
    state = s;
    handlers.state?.(s);
  };

  const open = () => {
    timer = null;
    setState("connecting");
    const es = create(url);
    source = es;
    for (const name of Object.keys(EVENT_SCHEMAS) as LiveEventName[]) {
      es.addEventListener(name, (ev) => dispatchEvent(name, String(ev.data ?? ""), handlers));
    }
    es.onopen = () => {
      attempt = 0;
      setState("open");
    };
    es.onerror = () => {
      if (source !== es || state === "closed") return;
      es.close();
      source = null;
      setState("paused");
      timer = setTimeout(open, backoffDelay(attempt++, minMs, maxMs));
    };
  };

  open();
  return {
    get state() {
      return state;
    },
    close() {
      setState("closed");
      if (timer) clearTimeout(timer);
      source?.close();
      source = null;
    },
  };
}
