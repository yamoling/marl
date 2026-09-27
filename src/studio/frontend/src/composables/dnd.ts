/**
 * Drag and drop of fields (metrics, parameters) and plots. The payload travels in the
 * DataTransfer (`application/x-marl-studio` and `text/plain`) and in a module-level ref, because
 * browsers hide DataTransfer data during `dragover`.
 */
import { ref } from "vue";

export type DragPayload =
  | { kind: "metric"; table: string; metric: string }
  | { kind: "param"; path: string }
  | { kind: "plot"; id: string };

export const MIME = "application/x-marl-studio";
export const dragging = ref<DragPayload | null>(null);

export function startDrag(ev: DragEvent, payload: DragPayload): void {
  dragging.value = payload;
  const json = JSON.stringify(payload);
  ev.dataTransfer?.setData(MIME, json);
  ev.dataTransfer?.setData("text/plain", json);
  if (ev.dataTransfer) ev.dataTransfer.effectAllowed = payload.kind === "plot" ? "move" : "copy";
}

export function endDrag(): void {
  dragging.value = null;
}

/** Validate an unknown value as a drag payload. @ai-generated */
export function asPayload(v: unknown): DragPayload | null {
  if (typeof v !== "object" || v === null) return null;
  const o = v as Record<string, unknown>;
  if (o.kind === "metric" && typeof o.table === "string" && typeof o.metric === "string") return { kind: "metric", table: o.table, metric: o.metric };
  if (o.kind === "param" && typeof o.path === "string") return { kind: "param", path: o.path };
  if (o.kind === "plot" && typeof o.id === "string") return { kind: "plot", id: o.id };
  return null;
}

/** Payload of a drop: the in-page drag, else the DataTransfer content. @ai-generated */
export function readDrop(ev: DragEvent): DragPayload | null {
  if (dragging.value) return dragging.value;
  const dt = ev.dataTransfer;
  if (!dt) return null;
  for (const type of [MIME, "text/plain"]) {
    try {
      const p = asPayload(JSON.parse(dt.getData(type)));
      if (p) return p;
    } catch {
      /* not ours */
    }
  }
  return null;
}

/** Whether the current drag (if any) is of one of `kinds`. */
export const isDragging = (...kinds: DragPayload["kind"][]): boolean => !!dragging.value && kinds.includes(dragging.value.kind);
