/**
 * Timeline tracks selected for replays, per experiment (port of the old UI's `TracksStore`, which
 * keyed them by logdir). Persisted in `localStorage["marl-studio.tracks"]`; entries that cannot be
 * read are dropped one by one instead of resetting everything.
 */
import { defineStore } from "pinia";
import { ref } from "vue";
import type { TrackKind } from "../domain/settings";
import type { TrackConfig } from "../domain/timeline";
import { parseJSON, readStorage, writeStorage } from "./storage";

export const TRACKS_KEY = "marl-studio.tracks";

type TrackMap = Record<string, TrackConfig[]>;

/** Tolerant parse of the stored map: invalid experiments or track entries are skipped. @ai-generated */
export function parseTracks(raw: unknown): TrackMap {
  const out: TrackMap = {};
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return out;
  for (const [exp, tracks] of Object.entries(raw)) {
    if (!Array.isArray(tracks)) continue;
    const ok = dedupe(
      tracks.flatMap((t): TrackConfig[] => {
        if (typeof t !== "object" || t === null) return [];
        const { label, kind } = t as { label?: unknown; kind?: unknown };
        if (typeof label !== "string" || !label) return [];
        return [{ label, kind: (kind === "categorical" ? "categorical" : "numeric") as TrackKind }];
      }),
    );
    if (ok.length) out[exp] = ok;
  }
  return out;
}

function dedupe(tracks: readonly TrackConfig[]): TrackConfig[] {
  const m = new Map<string, TrackConfig>();
  for (const t of tracks) m.set(t.label, { label: t.label, kind: t.kind });
  return [...m.values()];
}

export const useTracksStore = defineStore("tracks", () => {
  const selected = ref<TrackMap>(parseTracks(parseJSON(readStorage(TRACKS_KEY))));

  const forExperiment = (id: string): TrackConfig[] => selected.value[id] ?? [];

  function commit(id: string, tracks: TrackConfig[]): void {
    const next = { ...selected.value };
    if (tracks.length) next[id] = tracks;
    else delete next[id];
    selected.value = next;
    writeStorage(TRACKS_KEY, JSON.stringify(next));
  }

  /** Replace the selection of `id` (deduplicated by label, order kept). */
  function set(id: string, tracks: readonly TrackConfig[]): void {
    commit(id, dedupe(tracks));
  }

  function add(id: string, t: TrackConfig): void {
    const cur = forExperiment(id);
    if (!cur.some((x) => x.label === t.label)) commit(id, [...cur, { label: t.label, kind: t.kind }]);
  }

  function update(id: string, t: TrackConfig): void {
    const cur = forExperiment(id);
    if (cur.some((x) => x.label === t.label)) commit(id, cur.map((x) => (x.label === t.label ? { label: t.label, kind: t.kind } : x)));
  }

  function remove(id: string, label: string): void {
    commit(
      id,
      forExperiment(id).filter((x) => x.label !== label),
    );
  }

  /** Swap two tracks (reordering with the up/down buttons). @ai-generated */
  function swap(id: string, i: number, j: number): void {
    const cur = [...forExperiment(id)];
    if (i === j || i < 0 || j < 0 || i >= cur.length || j >= cur.length) return;
    [cur[i], cur[j]] = [cur[j], cur[i]];
    commit(id, cur);
  }

  return { selected, forExperiment, set, add, update, remove, swap };
});
