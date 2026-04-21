import { defineStore } from "pinia";
import { HTTP_URL } from "../constants";
import { Run, RunSchema } from "../models/Run";
import { ref } from "vue";
import { apiFetch, parseOrThrow } from "../api";

export const useRunStore = defineStore("RunStore", () => {
  const runs = ref(new Map<string, Run[]>());
  const loading = ref(new Map<string, boolean>());

  async function refresh(logdir: string) {
    loading.value.set(logdir, true);
    try {
      const updatedRuns = await getRuns(logdir);
      runs.value.set(logdir, updatedRuns);
    } finally {
      loading.value.set(logdir, false);
    }
  }

  async function getRuns(logdir: string) {
    const resp = await apiFetch(
      `${HTTP_URL}/runs/get/${logdir}`,
      undefined,
      "Failed to fetch runs",
    );
    return parseOrThrow(RunSchema.array(), await resp.json());
  }

  async function stopRun(logdir: string, rundir: string) {
    await apiFetch(
      `${HTTP_URL}/runs/stop/${rundir}`,
      { method: "POST" },
      "Failed to stop run",
    );
    await refresh(logdir);
  }

  async function startRun(
    logdir: string,
    rundir: string,
    device: string = "auto",
  ) {
    try {
      await apiFetch(
        `${HTTP_URL}/runs/start/${rundir}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ device }),
        },
        "Failed to start run",
      );
      await refresh(logdir);
    } catch {
      /* already toasted */
    }
  }

  async function remove(logdir: string) {
    runs.value.delete(logdir);
  }

  return {
    runs,
    loading,
    getRuns,
    refresh,
    stopRun,
    startRun,
    remove,
  };
});
