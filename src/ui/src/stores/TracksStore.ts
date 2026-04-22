import { defineStore } from "pinia";
import { ref } from "vue";
import {
  TrackConfig,
  type TimelineTrackKind,
  TrackConfigSchema,
  Track,
  TrackGroup,
} from "../models/Timeline";
import { useErrorStore } from "./ErrorStore";

const STORAGE_KEY = "tracksStore";
const SEEN_TRACKS_STORAGE_KEY = "tracksStore_seen_labels";

/**
 * The TracksStore is responsible for managing the tracks that the user has selected to display for each logdir.
 * It persists the selections in localStorage to maintain them across sessions.
 */
export const useTracksStore = defineStore("TracksStore", () => {
  const errorStore = useErrorStore();
  const selectedTracks = ref(load());
  const allTrackLabels = ref(loadSeenTrackLabels());

  function load(): { [logdir: string]: TrackConfig[] } {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw == null) {
      return {};
    }
    try {
      const parsed: unknown = JSON.parse(raw);
      if (
        typeof parsed !== "object" ||
        parsed === null ||
        Array.isArray(parsed)
      ) {
        throw new Error("Expected a plain object");
      }
      const result: { [logdir: string]: TrackConfig[] } = {};
      for (const [logdir, tracks] of Object.entries(
        parsed as Record<string, unknown>,
      )) {
        const trackResult = TrackConfigSchema.array().safeParse(tracks);
        if (!trackResult.success) {
          throw new Error(`Invalid tracks for logdir "${logdir}"`);
        }
        result[logdir] = trackResult.data;
      }
      return result;
    } catch {
      errorStore.push(
        "Track selection reset",
        "Your saved track selections could not be read and have been reset to defaults.",
      );
      return {};
    }
  }

  function loadSeenTrackLabels(): Set<string> {
    const raw = window.localStorage.getItem(SEEN_TRACKS_STORAGE_KEY);
    if (raw == null) {
      return new Set<string>();
    }
    try {
      const parsed: unknown = JSON.parse(raw);
      if (!Array.isArray(parsed)) {
        return new Set<string>();
      }
      const labels = parsed
        .filter((value): value is string => typeof value === "string")
        .map((label) => label.trim())
        .filter((label) => label.length > 0);
      return new Set(labels);
    } catch {
      return new Set<string>();
    }
  }

  function forLogdir(logdir: string): TrackConfig[] {
    return selectedTracks.value[logdir] ?? [];
  }

  function commit(nextMap: { [logdir: string]: TrackConfig[] }) {
    selectedTracks.value = nextMap;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(selectedTracks.value));
  }

  function add(logdir: string, track: TrackConfig) {
    const tracks = forLogdir(logdir);
    const existing = tracks.find((entry) => entry.label === track.label);
    if (existing !== undefined) {
      return;
    }
    const nextMap = { ...selectedTracks.value };
    nextMap[logdir] = [...tracks, { label: track.label, kind: track.kind }];
    commit(nextMap);
  }

  function update(logdir: string, track: TrackConfig) {
    const tracks = forLogdir(logdir);
    const trackIndex = tracks.findIndex((entry) => entry.label === track.label);
    if (trackIndex === -1) return;

    const nextTracks = [...tracks];
    nextTracks[trackIndex] = { label: track.label, kind: track.kind };
    const nextMap = { ...selectedTracks.value };
    nextMap[logdir] = nextTracks;
    commit(nextMap);
  }

  function set(logdir: string, tracks: TrackConfig[]) {
    const nextMap = { ...selectedTracks.value };
    if (tracks.length === 0) {
      delete nextMap[logdir];
    } else {
      const deduplicated = new Map<string, TrackConfig>();
      for (const track of tracks) {
        deduplicated.set(track.label, { label: track.label, kind: track.kind });
      }
      nextMap[logdir] = Array.from(deduplicated.values());
    }
    commit(nextMap);
  }

  function remove(logdir: string, track: TrackConfig) {
    const tracks = forLogdir(logdir);
    if (tracks.length === 0) return;

    const nextTracks = tracks.filter((entry) => entry.label !== track.label);
    const nextMap = { ...selectedTracks.value };
    if (nextTracks.length === 0) {
      delete nextMap[logdir];
    } else {
      nextMap[logdir] = nextTracks;
    }
    commit(nextMap);
  }

  function swap(logdir: string, index1: number, index2: number) {
    const tracks = forLogdir(logdir);
    if (tracks.length === 0) return;

    if (
      index1 < 0 ||
      index1 >= tracks.length ||
      index2 < 0 ||
      index2 >= tracks.length
    )
      return;
    if (index1 === index2) return;

    const nextTracks = [...tracks];
    const tmp = nextTracks[index1];
    nextTracks[index1] = nextTracks[index2];
    nextTracks[index2] = tmp;
    const nextMap = { ...selectedTracks.value };
    nextMap[logdir] = nextTracks;
    commit(nextMap);
  }

  function notifyTracks(tracks: (Track | TrackGroup)[]) {
    for (const track of tracks) {
      allTrackLabels.value.add(track.label);
    }
    localStorage.setItem(
      SEEN_TRACKS_STORAGE_KEY,
      JSON.stringify(Array.from(allTrackLabels.value).sort()),
    );
  }

  return {
    selectedTracks,
    allTrackLabels,
    set,
    add,
    remove,
    swap,
    update,
    forLogdir,
    notifyTracks,
  };
});
