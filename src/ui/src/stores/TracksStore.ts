import { defineStore } from 'pinia';
import { ref } from 'vue';
import { TrackConfig, type TimelineTrackKind, TrackConfigSchema } from '../models/Timeline';
const STORAGE_KEY = 'tracksStore';



/**
 * The TracksStore is responsible for managing the tracks that the user has selected to display for each logdir. 
 * It persists the selections in localStorage to maintain them across sessions.
 */
export const useTracksStore = defineStore('TracksStore', () => {
    const selectedTracks = ref(load());

    function commit(nextMap: { [logdir: string]: TrackConfig[] }) {
        selectedTracks.value = nextMap;
        save();
    }

    function add(logdir: string, track: TrackConfig) {
        const tracks = selectedTracks.value[logdir] ?? [];
        const existing = tracks.find((entry) => entry.label === track.label);
        if (existing !== undefined) {
            return
        }
        const nextMap = { ...selectedTracks.value };
        nextMap[logdir] = [...tracks, { label: track.label, kind: track.kind }];
        commit(nextMap);
    }

    function update(logdir: string, track: TrackConfig) {
        const tracks = selectedTracks.value[logdir];
        if (tracks == null) return;

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
        const tracks = selectedTracks.value[logdir];
        if (tracks == null) return;

        const nextTracks = tracks.filter((entry) => entry.label !== track.label);
        const nextMap = { ...selectedTracks.value };
        if (nextTracks.length === 0) {
            delete nextMap[logdir];
        } else {
            nextMap[logdir] = nextTracks;
        }
        commit(nextMap);
    }

    function save() {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(selectedTracks.value));
    }


    function swap(logdir: string, index1: number, index2: number) {
        const tracks = selectedTracks.value[logdir];
        if (tracks == null) return;

        if (index1 < 0 || index1 >= tracks.length || index2 < 0 || index2 >= tracks.length) return;
        if (index1 === -1 || index2 === -1) return;

        const tmp = tracks[index1];
        tracks[index1] = tracks[index2];
        tracks[index2] = tmp;
        const nextMap = { ...selectedTracks.value };
        nextMap[logdir] = [...tracks];
        commit(nextMap);
    }

    return {
        selectedTracks,
        set,
        add,
        remove,
        swap,
        update
    };
});

function load(): { [logdir: string]: TrackConfig[] } {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw == null) {
        return {};
    }
    try {
        const parsed = JSON.parse(raw) as { [logdir: string]: { label: string; kind: TimelineTrackKind }[] };
        return Object.fromEntries(Object.entries(parsed).map(([logdir, tracks]) => [logdir, TrackConfigSchema.array().catch([]).parse(tracks)]))
    } catch (e) {
        return {};
    }

}



