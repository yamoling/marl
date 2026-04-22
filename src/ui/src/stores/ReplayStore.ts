import { defineStore } from "pinia";
import { ReplayEpisode } from "../models/Episode";
import { HTTP_URL } from "../constants";
import { apiFetch } from "../api";
import { ref } from "vue";

export const useReplayStore = defineStore("ReplayStore", () => {
  const loading = ref(false);

  async function getEpisode(
    test_step: number,
    test_num: number,
    rundir: string,
    only_saved_actions: boolean,
  ) {
    loading.value = true;
    try {
      const resp = await apiFetch(`${HTTP_URL}/experiment/replay/${test_step}/${test_num}/${only_saved_actions}/${rundir}`);
      const json = await resp.json();
      return ReplayEpisode.fromJSON(json);
    } finally {
      loading.value = false;
    }
  }

  return { getEpisode, loading };
});
