import { ref } from "vue";
import { defineStore } from "pinia";
import { ReplayEpisode, ReplayEpisodeSummary } from "../models/Episode";
import { HTTP_URL } from "../constants";

export const useReplayStore = defineStore("ReplayStore", () => {

    const logdirs = ref([] as string[]);

    function refresh() {
        fetch(`${HTTP_URL}/ls/logs`)
            .then(resp => resp.json())
            .then(d => {
                const dirs = d as any[];
                logdirs.value = dirs.map(d => d.path);
            });

    }
    refresh();

    async function getEpisode(directory: string): Promise<ReplayEpisode> {
        const resp = await fetch(`${HTTP_URL}/replay/episode/${directory}`);
        return await resp.json();
    }

    async function getTestEpisodes(directory: string): Promise<ReplayEpisodeSummary[]> {
        const resp = await fetch(`${HTTP_URL}/replay/tests/summary/${directory}`);
        return await resp.json();
    }


    return { logdirs, getEpisode, getTestEpisodes };
});
