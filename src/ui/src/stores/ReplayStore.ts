import { defineStore } from "pinia";
import { ReplayEpisode } from "../models/Episode";
import { HTTP_URL } from "../constants";
import { apiFetch } from "../api";

export const useReplayStore = defineStore("ReplayStore", () => {
  async function getEpisode(directory: string) {
    const resp = await apiFetch(
      `${HTTP_URL}/experiment/replay/${directory}`,
      undefined,
      undefined,
      true, // silent: EpisodeReplay.vue handles the error inline
    );
    const json = await resp.json();
    return ReplayEpisode.fromJSON(json);
  }

  return { getEpisode };
});
