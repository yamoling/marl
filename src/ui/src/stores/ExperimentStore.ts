import { defineStore } from "pinia";
import { HTTP_URL } from "../constants";
import { Experiment, ExperimentSchema } from "../models/Experiment";
import { ReplayEpisodeSummarySchema } from "../models/Episode";
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
  const trainerNames = computed(() => experiments.value.map(exp => exp.trainer.name));

  async function loadExperiments(): Promise<Experiment[]> {
    try {
      loading.value = true;
      const resp = await apiFetch(`${HTTP_URL}/experiment/list`, "Failed to load experiments");
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
    const resp = await apiFetch(`${HTTP_URL}/experiment/${logdir}`, "Failed to load experiment");
    return parseOrThrow(ExperimentSchema, await resp.json());
  }

  /**
   * Ask the backend to load an experiment, which is required to
   * replay an episode.
   * @param logdir
   */
  async function loadExperiment(logdir: string) {
    await apiFetch(`${HTTP_URL}/experiment/load/${logdir}`, "Failed to pre-load experiment", { method: "POST" });
  }

  async function unloadExperiment(logdir: string) {
    await fetch(`${HTTP_URL}/experiment/load/${logdir}`, { method: "DELETE" });
    // errors silently ignored — this is cleanup
  }

  async function getTestEpisodes(logdir: string, time_step: number) {
    const resp = await apiFetch(`${HTTP_URL}/experiment/test/list/${time_step}/${logdir}`, "Failed to fetch test episodes");
    const json = await resp.json();
    return ReplayEpisodeSummarySchema.array().parse(json);
  }

  async function rename(logdir: string, newLogdir: string) {
    await apiFetch(`${HTTP_URL}/experiment/rename`, "Failed to rename experiment", jsonBody({ logdir, newLogdir }));
    const exp = experiments.value.find((exp) => exp.logdir === logdir);
    if (exp) {
      exp.logdir = newLogdir;
    }
  }

  async function remove(logdir: string) {
    await apiFetch(`${HTTP_URL}/experiment/delete/${logdir}`, "Failed to delete experiment", { method: "DELETE" });
    experiments.value = experiments.value.filter(
      (exp) => exp.logdir !== logdir,
    );
    runStore.remove(logdir);
  }

  async function stopRuns(logdir: string) {
    await apiFetch(`${HTTP_URL}/experiment/stop-runs/${logdir}`, "Failed to stop runs", { method: "POST" });
    await runStore.refresh(logdir);
  }

  async function getEnvImage(logdir: string, seed: number): Promise<string> {
    const resp = await apiFetch(`${HTTP_URL}/experiment/image/${seed}/${logdir}`, "Failed to load environment image");
    return await resp.text();
  }

  async function newRun(
    logdir: string,
    nRuns: number,
    nJobs: number,
    seed: number,
    nTests: number,
    gpuStrategy: "scatter" | "group" = "scatter",
    disabledDevices: number[] = [],
  ) {
    await apiFetch(`${HTTP_URL}/runner/new/${logdir}`, "Failed to start new run", jsonBody({ seed, nTests, nRuns, nJobs, gpuStrategy, disabledDevices }));
    await runStore.refresh(logdir);
  }

  async function refresh() {
    experiments.value = await loadExperiments();
  }


  return {
    loading,
    experiments,
    isRunning,
    trainerNames,
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
