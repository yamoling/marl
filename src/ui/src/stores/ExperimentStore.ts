import { defineStore } from "pinia";
import { HTTP_URL } from "../constants";
import { Experiment, ExperimentSchema } from "../models/Experiment";
import { ReplayEpisodeSummary } from "../models/Episode";
import { computed, ref } from "vue";
import { apiFetch, jsonBody, parseOrThrow } from "../api";
import { useRunStore } from "./RunStore";

export const useExperimentStore = defineStore("ExperimentStore", () => {
  const loading = ref(false);
  const experiments = ref([] as Experiment[]);
  const runStore = useRunStore();
  const isRunning = computed(() => {
    const res = {} as { [logdir: string]: boolean };
    runStore.runs.forEach((runs, logdir) => {
      res[logdir] = runs.some((run) => run.pid != null);
    });
    return res;
  });
  refresh();

  async function loadExperiments(): Promise<Experiment[]> {
    try {
      loading.value = true;
      const resp = await apiFetch(
        `${HTTP_URL}/experiment/list`,
        { headers: { "Access-Control-Allow-Origin": "*" } },
        "Failed to load experiments",
      );
      const json = await resp.json();
      const experiments = parseOrThrow(ExperimentSchema.array(), json);
      for (const exp of experiments) {
        runStore.refresh(exp.logdir);
      }
      return experiments;
    } catch {
      return [];
    } finally {
      loading.value = false;
    }
  }

  async function getExperiment(logdir: string) {
    const resp = await apiFetch(
      `${HTTP_URL}/experiment/${logdir}`,
      undefined,
      "Failed to load experiment",
    );
    return parseOrThrow(ExperimentSchema, await resp.json());
  }

  /**
   * Ask the backend to load an experiment, which is required to
   * replay an episode.
   * @param logdir
   */
  async function loadExperiment(logdir: string) {
    await apiFetch(
      `${HTTP_URL}/experiment/load/${logdir}`,
      { method: "POST" },
      "Failed to pre-load experiment",
    );
  }

  async function unloadExperiment(logdir: string) {
    await fetch(`${HTTP_URL}/experiment/load/${logdir}`, { method: "DELETE" });
    // errors silently ignored — this is cleanup
  }

  async function getTestEpisodes(
    logdir: string,
    time_step: number,
  ): Promise<ReplayEpisodeSummary[]> {
    try {
      const resp = await apiFetch(
        `${HTTP_URL}/experiment/test/list/${time_step}/${logdir}`,
        undefined,
        "Failed to fetch test episodes",
      );
      return await resp.json();
    } catch {
      return [];
    }
  }

  async function rename(logdir: string, newLogdir: string) {
    try {
      await apiFetch(
        `${HTTP_URL}/experiment/rename`,
        jsonBody({ logdir, newLogdir }),
        "Failed to rename experiment",
      );
      experiments.value = await loadExperiments();
    } catch {
      /* already toasted */
    }
  }

  async function remove(logdir: string) {
    try {
      await apiFetch(
        `${HTTP_URL}/experiment/delete/${logdir}`,
        { method: "DELETE" },
        "Failed to delete experiment",
      );
      experiments.value = experiments.value.filter(
        (exp) => exp.logdir !== logdir,
      );
      runStore.remove(logdir);
    } catch {
      /* already toasted */
    }
  }

  async function stopRuns(logdir: string) {
    try {
      await apiFetch(
        `${HTTP_URL}/experiment/stop-runs/${logdir}`,
        { method: "POST" },
        "Failed to stop runs",
      );
      await runStore.refresh(logdir);
    } catch {
      /* already toasted */
    }
  }

  async function getEnvImage(logdir: string, seed: number): Promise<string> {
    try {
      const resp = await apiFetch(
        `${HTTP_URL}/experiment/image/${seed}/${logdir}`,
        undefined,
        "Failed to load environment image",
      );
      return await resp.text();
    } catch {
      return "";
    }
  }

  async function newRun(
    logdir: string,
    nRuns: number,
    seed: number,
    nTests: number,
    gpuStrategy: "scatter" | "group" = "scatter",
    disabledDevices: number[] = [],
  ) {
    try {
      await apiFetch(
        `${HTTP_URL}/runner/new/${logdir}`,
        jsonBody({ seed, nTests, nRuns, gpuStrategy, disabledDevices }),
        "Failed to start new run",
      );
      await runStore.refresh(logdir);
      return true;
    } catch {
      return false;
    }
  }

  async function refresh() {
    experiments.value = await loadExperiments();
  }

  return {
    loading,
    experiments,
    isRunning,
    refresh,
    getExperiment,
    loadExperiment,
    unloadExperiment,
    getTestEpisodes,
    remove,
    stopRuns,
    rename,
    getEnvImage,
    newRun,
  };
});
