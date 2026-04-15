import { defineStore } from 'pinia';
import { ref } from 'vue';
import { TrackConfig, type TimelineTrackKind } from '../models/Timeline';

const STORAGE_KEY = 'tracksStore';



/**
 * The TracksStore is responsible for managing the tracks that the user has selected to display for each logdir. 
 * It persists the selections in localStorage to maintain them across sessions.
 */
export const useTracksStore = defineStore('TracksStore', () => {
    const selectedTracks = ref(load());

    function commit(nextMap: Map<string, TrackConfig[]>) {
        selectedTracks.value = new Map(nextMap);
        save();
    }

    function add(logdir: string, track: TrackConfig) {
        const tracks = selectedTracks.value.get(logdir) ?? [];
        const existing = tracks.find((entry) => entry.label === track.label);
        if (existing !== undefined) {
            return
        }
        const nextMap = new Map(selectedTracks.value);
        nextMap.set(logdir, [...tracks, { label: track.label, kind: track.kind }]);
        commit(nextMap);
    }

    function update(logdir: string, track: TrackConfig) {
        const tracks = selectedTracks.value.get(logdir);
        if (tracks == null) return;

        const trackIndex = tracks.findIndex((entry) => entry.label === track.label);
        if (trackIndex === -1) return;

        const nextTracks = [...tracks];
        nextTracks[trackIndex] = { label: track.label, kind: track.kind };
        const nextMap = new Map(selectedTracks.value);
        nextMap.set(logdir, nextTracks);
        commit(nextMap);
    }

    function set(logdir: string, tracks: TrackConfig[]) {
        const nextMap = new Map(selectedTracks.value);
        if (tracks.length === 0) {
            nextMap.delete(logdir);
            commit(nextMap);
            return;
        }

        const deduplicated = new Map<string, TrackConfig>();
        for (const track of tracks) {
            deduplicated.set(track.label, { label: track.label, kind: track.kind });
        }
        nextMap.set(logdir, Array.from(deduplicated.values()));
        commit(nextMap);
    }

    function remove(logdir: string, track: TrackConfig) {
        const tracks = selectedTracks.value.get(logdir);
        if (tracks == null) return;

        const nextTracks = tracks.filter((entry) => entry.label !== track.label);
        const nextMap = new Map(selectedTracks.value);
        if (nextTracks.length === 0) {
            nextMap.delete(logdir);
        } else {
            nextMap.set(logdir, nextTracks);
        }
        commit(nextMap);
    }

    function save() {
        // TODO: only keep the label and kind of the tracks to reduce the size of the stored data
        const serializable = Object.fromEntries(Array.from(selectedTracks.value.entries()).map(([logdir, tracks]) => [logdir, tracks.map((track) => ({ label: track.label, kind: track.kind }))]));
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(serializable));
    }

    function get(logdir: string): TrackConfig[] {
        return selectedTracks.value.get(logdir) ?? [];
    }

    function swap(logdir: string, index1: number, index2: number) {
        const tracks = selectedTracks.value.get(logdir);
        if (tracks == null) return;

        if (index1 < 0 || index1 >= tracks.length || index2 < 0 || index2 >= tracks.length) return;
        if (index1 === -1 || index2 === -1) return;

        const tmp = tracks[index1];
        tracks[index1] = tracks[index2];
        tracks[index2] = tmp;
        const nextMap = new Map(selectedTracks.value);
        nextMap.set(logdir, [...tracks]);
        commit(nextMap);
    }

    return {
        get,
        set,
        add,
        remove,
        swap,
        update
    };
});

function load(): Map<string, TrackConfig[]> {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw == null) return new Map();

    const parsed = JSON.parse(raw) as { [logdir: string]: { label: string; kind: TimelineTrackKind }[] };
    return new Map(Object.entries(parsed).map(([logdir, tracks]) => [logdir, tracks.map((track) => ({ label: track.label, kind: track.kind }))]));
}



