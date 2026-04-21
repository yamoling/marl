import { z } from "zod";
import { useErrorStore } from "./stores/ErrorStore";

interface ApiErrorBody {
  error?: string;
  message?: string;
}

async function parseErrorBody(resp: Response): Promise<ApiErrorBody> {
  // Clone so body can be read once
  const text = await resp.text();
  try {
    const json = JSON.parse(text);
    if (typeof json === "object" && json !== null) {
      return {
        error: typeof json.error === "string" ? json.error : "",
        message: typeof json.message === "string" ? json.message : text,
      };
    }
  } catch {
    // not JSON
  }
  return { error: "", message: text || `HTTP ${resp.status}` };
}

/**
 * Wrapper around fetch that:
 * - On network error: pushes a toast to ErrorStore and re-throws.
 * - On HTTP error (!resp.ok): reads the body, pushes a toast, and throws Error(message).
 * - On success: returns the Response with body untouched.
 *
 * @param url        - The URL to fetch.
 * @param errorTitle - Human-friendly title shown in the toast (e.g. "Failed to start run").
 * @param options    - Standard RequestInit options.
 */
export async function apiFetch(
  url: string,
  errorTitle?: string,
  options?: RequestInit,
): Promise<Response> {
  let resp: Response;
  try {
    resp = await fetch(url, options);
  } catch (networkError) {
    const message =
      networkError instanceof Error
        ? networkError.message
        : String(networkError);
    const errorStore = useErrorStore();
    errorStore.push(errorTitle ?? "Network error", message, "NetworkError");
    throw networkError;
  }

  if (!resp.ok) {
    const body = await parseErrorBody(resp);
    const detail = body.message ?? `HTTP ${resp.status}`;
    const errorType = body.error ?? "";
    const errorStore = useErrorStore();
    errorStore.push(errorTitle ?? "Request failed", detail, errorType);
    throw new Error(detail);
  }
  return resp;
}

/** Builds a RequestInit for a JSON POST/PUT/PATCH/DELETE body. */
export function jsonBody(data: object, method: string = "POST"): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  };
}

/**
 * Parses `data` with the given Zod schema. If parsing fails, formats the
 * ZodError issues into a human-readable message, pushes it to ErrorStore,
 * and re-throws — mirroring the behaviour of `apiFetch` for HTTP errors.
 *
 * The outer `catch {}` in the caller still handles the throw and returns a
 * default value; the toast is already queued before the throw.
 *
 * @param schema     - The Zod schema to validate against.
 * @param data       - The raw value to parse (typically `await resp.json()`).
 */
export function parseOrThrow<T>(schema: z.ZodType<T>, data: unknown): T {
  const result = schema.safeParse(data);
  if (result.success) {
    return result.data;
  }

  const lines = result.error.issues.map((issue) => {
    const path = issue.path.length > 0 ? issue.path.join(".") : "(root)";
    return `  • ${path}: ${issue.message}`;
  });
  const detail = `The server returned data in an unexpected format:\n${lines.join("\n")}`;

  const errorStore = useErrorStore();
  errorStore.push("Parsing error", detail, "ZodError");

  throw result.error;
}
