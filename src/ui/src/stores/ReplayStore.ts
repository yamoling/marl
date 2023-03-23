import { defineStore } from "pinia";
import { ReplayEpisode, ReplayEpisodeSummary } from "../models/Episode";
import { HTTP_URL } from "../constants";

export const useReplayStore = defineStore("ReplayStore", () => {
    async function getEpisode(directory: string): Promise<ReplayEpisode> {
        const resp = await fetch(`${HTTP_URL}/replay/episode/${directory}`);
        return await resp.json();
    }

    async function getTestEpisodes(directory: string): Promise<ReplayEpisodeSummary[]> {
        const resp = await fetch(`${HTTP_URL}/replay/test/list/${directory}`);
        return await resp.json();
    }

    return { getEpisode, getTestEpisodes };
});
