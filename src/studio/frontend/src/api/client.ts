/**
 * Typed `fetch` wrapper for the Studio API: JSON in and out, `ErrorBody` → `ApiError`,
 * cancellation through `AbortSignal`, and response validation with Zod schemas.
 */
import type { z } from "zod";
import { ErrorBodySchema, formatZodError, type Issue } from "./schemas";

export const API_BASE = "/api";

/** Error raised for any failed API call (HTTP error, network error or unreadable response). */
export class ApiError extends Error {
  /** HTTP status; 0 for network errors, -1 for responses that failed validation. */
  readonly status: number;
  /** `error` field of the ErrorBody, or `"network"` / `"invalid-response"` / `"http-<status>"`. */
  readonly code: string;
  readonly issue: Issue | undefined;
  readonly body: unknown;

  constructor(status: number, code: string, message: string, issue?: Issue, body?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.issue = issue;
    this.body = body;
  }
}

/** True when `e` is the rejection of an aborted request. @ai-generated */
export function isAbortError(e: unknown): boolean {
  return e instanceof DOMException ? e.name === "AbortError" : (e as { name?: string } | null)?.name === "AbortError";
}

/** Encode an ExperimentId or RunId for a URL: each path segment is encoded separately. @ai-generated */
export function encodeId(id: string): string {
  return id.split("/").map(encodeURIComponent).join("/");
}

export type QueryValue = string | number | boolean | null | undefined;

export type RequestOptions<S extends z.ZodType | undefined> = {
  body?: unknown;
  query?: Record<string, QueryValue>;
  signal?: AbortSignal;
  schema?: S;
  fetchImpl?: typeof fetch;
};

type Result<S> = S extends z.ZodType ? z.output<S> : undefined;

/** Build `/api/<path>?<query>`, skipping null/undefined/empty query values. @ai-generated */
export function buildUrl(path: string, query?: Record<string, QueryValue>): string {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(query ?? {})) {
    if (v === null || v === undefined || v === "") continue;
    qs.set(k, String(v));
  }
  const s = qs.toString();
  return `${API_BASE}${path.startsWith("/") ? path : "/" + path}${s ? "?" + s : ""}`;
}

/**
 * Perform one API request. On success the JSON body is validated with `schema` (when given) and
 * returned; without a schema, or on `204`, the result is `undefined`. Non-2xx responses throw an
 * `ApiError` built from the contract's `ErrorBody` when it can be read.
 *
 * @ai-generated
 */
export async function request<S extends z.ZodType | undefined = undefined>(
  method: "GET" | "POST" | "PATCH" | "DELETE",
  path: string,
  opts: RequestOptions<S> = {},
): Promise<Result<S>> {
  const url = buildUrl(path, opts.query);
  const init: RequestInit = { method, signal: opts.signal, headers: { Accept: "application/json" } };
  if (opts.body !== undefined) {
    init.body = JSON.stringify(opts.body);
    init.headers = { ...init.headers, "Content-Type": "application/json" };
  }
  let res: Response;
  try {
    res = await (opts.fetchImpl ?? fetch)(url, init);
  } catch (e) {
    if (isAbortError(e)) throw e;
    throw new ApiError(0, "network", `Network error on ${method} ${url}: ${(e as Error)?.message ?? e}`);
  }
  const text = await res.text();
  let json: unknown = undefined;
  if (text) {
    try {
      json = JSON.parse(text);
    } catch {
      if (res.ok) throw new ApiError(-1, "invalid-response", `${method} ${url} returned invalid JSON`, undefined, text);
    }
  }
  if (!res.ok) {
    const body = ErrorBodySchema.safeParse(json);
    if (body.success) throw new ApiError(res.status, body.data.error, body.data.message || body.data.error, body.data.issue, json);
    throw new ApiError(res.status, `http-${res.status}`, `${method} ${url} failed with HTTP ${res.status}`, undefined, json ?? text);
  }
  if (!opts.schema || res.status === 204) return undefined as Result<S>;
  const parsed = opts.schema.safeParse(json);
  if (!parsed.success) {
    throw new ApiError(-1, "invalid-response", `${method} ${url}: unexpected response (${formatZodError(parsed.error)})`, undefined, json);
  }
  return parsed.data as Result<S>;
}
