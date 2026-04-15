import { defineStore } from "pinia";
import { ReplayEpisode } from "../models/Episode";
import { HTTP_URL } from "../constants";

export const useReplayStore = defineStore("ReplayStore", () => {
    async function getEpisode(directory: string) {
        const resp = await fetch(`${HTTP_URL}/experiment/replay/${directory}`);
        const json = await resp.json();
        return ReplayEpisode.fromJSON(json);
    }

    return { getEpisode };
});
