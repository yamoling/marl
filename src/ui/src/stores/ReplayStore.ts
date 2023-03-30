import { defineStore } from "pinia";
import { ReplayEpisode } from "../models/Episode";
import { HTTP_URL } from "../constants";

export const useReplayStore = defineStore("ReplayStore", () => {
    async function getEpisode(directory: string): Promise<ReplayEpisode> {
        const resp = await fetch(`${HTTP_URL}/replay/episode/${directory}`);
        return await resp.json();
    }

    return { getEpisode };
});
