import { ref } from "vue";
import { defineStore } from "pinia";
import { ReplayEpisode, ReplayEpisodeSummary } from "../models/Episode";
import { HTTP_URL } from "../constants";
import { Experiment } from "../models/Experiment";

export const useReplayStore = defineStore("ReplayStore", () => {

    const logdirs = ref([] as string[]);
    const experiments = ref([] as Experiment[]);

    function refresh() {
        fetch(`${HTTP_URL}/ls/logs`)
            .then(resp => resp.json())
            .then((dirs: { path: string }[]) => {
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

    async function getExperiments(): Promise<Experiment[]> {
        const resp = await fetch(`${HTTP_URL}/experiments/list`);
        return await resp.json();
    }


    return { logdirs, getEpisode, getTestEpisodes, getExperiments };
});
