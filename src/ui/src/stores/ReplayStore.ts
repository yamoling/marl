import { defineStore } from "pinia";
import { ReplayEpisode } from "../models/Episode";
import { HTTP_URL } from "../constants";
import { ActionSpace } from "../models/Env";

export const useReplayStore = defineStore("ReplayStore", () => {
    async function getEpisode(directory: string): Promise<ReplayEpisode> {
        const resp = await fetch(`${HTTP_URL}/experiment/replay/${directory}`);
        const replay = await resp.json() as ReplayEpisode;

        if (replay.action_space != null && replay.action_space.space_type == null) {
            const hasBounds = Array.isArray((replay.action_space as { low?: unknown }).low)
                || Array.isArray((replay.action_space as { high?: unknown }).high);
            replay.action_space = {
                ...replay.action_space,
                space_type: hasBounds ? "continuous" : "discrete",
            } as ActionSpace;
        }

        return replay;
    }

    return { getEpisode };
});
