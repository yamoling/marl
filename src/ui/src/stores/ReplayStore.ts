import { defineStore } from "pinia";
import { AgentDetailsSchema, ReplayEpisode, ReplayEpisodeSchema } from "../models/Episode";
import { HTTP_URL } from "../constants";
import { apiFetch, parseOrThrow } from "../api";
import { ref } from "vue";

export const useReplayStore = defineStore("ReplayStore", () => {
  const loading = ref(false);

  async function getEpisode(test_step: number, test_num: number, rundir: string, only_saved_actions: boolean) {
    loading.value = true;
    try {
      const resp = await apiFetch(
        `${HTTP_URL}/experiment/replay/${test_step}/${test_num}/${only_saved_actions}/${rundir}`,
        "Failed to load replay episode",
      );
      const json = await resp.json();
      const payload = parseOrThrow(ReplayEpisodeSchema, json);
      console.debug(payload);
      return ReplayEpisode.fromJSON(payload);
    } finally {
      loading.value = false;
    }
  }

  return { getEpisode, loading };
});
