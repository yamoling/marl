import { defineStore } from "pinia";
import { Dataset, ExperimentResults } from "../models/Experiment";
import { ReplayEpisodeSummarySchema } from "../models/Episode";
import { ref, watch } from "vue";
import { apiFetch } from "../api";
import { HTTP_URL } from "../constants";
import { useSettingsStore } from "./SettingsStore";

export const useResultsStore = defineStore("ResultsStore", () => {
  const settingsStore = useSettingsStore();
  const results = ref(new Map<string, ExperimentResults>());
  const loading = ref(new Map<string, boolean>());
  const granularity = ref<number | null>(null);

  function normalizeGranularity(value: number): number {
    if (!Number.isFinite(value)) {
      throw new Error(`Invalid granularity value: ${value}`);
    }
    return Math.max(1, Math.round(value));
  }

  function currentGranularity(defaultGranularity?: number): number {
    if (granularity.value == null) {
      if (defaultGranularity == null) {
        throw new Error("Granularity must be set before loading results");
      }
      granularity.value = normalizeGranularity(defaultGranularity);
    }
    return granularity.value;
  }

  async function load(
    logdir: string,
    defaultGranularity?: number,
  ): Promise<ExperimentResults> {
    const activeGranularity = currentGranularity(defaultGranularity);
    const useWallTime = settingsStore.settings.visualization.useWallTime;
    loading.value.set(logdir, true);
    try {
      const resp = await apiFetch(
        `${HTTP_URL}/results/load/${logdir}?granularity=${activeGranularity}&use_wall_time=${useWallTime}`,
        `Failed to load results for ${logdir}`,
      );
      const datasets = (await resp.json()) as Dataset[];
      const experimentResults = new ExperimentResults(logdir, datasets);
      results.value.set(logdir, experimentResults);
      return experimentResults;
    } finally {
      loading.value.set(logdir, false);
      // Note: if apiFetch throws, the finally still runs (clears loading), then the error propagates to the caller.
    }
  }

  function unload(logdir: string) {
    results.value.delete(logdir);
  }

  async function reloadLoadedResults() {
    if (granularity.value == null) {
      return;
    }
    await Promise.all(
      Array.from(results.value.keys()).map((logdir) =>
        load(logdir, granularity.value ?? undefined),
      ),
    );
  }

  /**
   * Get the unagregated test results for a given experiment at a given time step.
   */
  async function getTestsResultsAt(logdir: string, timeStep: number) {
    const resp = await apiFetch(
      `${HTTP_URL}/results/test/${timeStep}/${logdir}`,
      `Failed to fetch test results at step ${timeStep}`,
    );
    const json = await resp.json();
    return ReplayEpisodeSummarySchema.array().parse(json);

  }

  async function getResultsByRun(logdir: string): Promise<ExperimentResults[]> {
    try {
      const activeGranularity = granularity.value;
      const useWallTime = settingsStore.settings.visualization.useWallTime;
      const granularityQuery =
        activeGranularity == null ? "" : `?granularity=${activeGranularity}`;
      const wallTimeQuery =
        granularityQuery.length === 0
          ? `?use_wall_time=${useWallTime}`
          : `&use_wall_time=${useWallTime}`;
      const resp = await apiFetch(
        `${HTTP_URL}/results/load-by-run/${logdir}${granularityQuery}${wallTimeQuery}`,
        `Failed to load per-run results for ${logdir}`,
      );
      const datasets = (await resp.json()) as Dataset[][];
      return datasets
        .filter((ds) => ds.length > 0)
        .map((ds) => new ExperimentResults(ds[0].logdir, ds));
    } catch {
      return [];
    }
  }

  function isLoaded(logdir: string): boolean {
    return results.value.has(logdir);
  }

  watch(granularity, async (newGranularity, oldGranularity) => {
    if (newGranularity !== oldGranularity) {
      await reloadLoadedResults();
    }
  });

  watch(
    () => settingsStore.settings.visualization.useWallTime,
    async (newUseWallTime, oldUseWallTime) => {
      if (newUseWallTime !== oldUseWallTime) {
        await reloadLoadedResults();
      }
    },
  );

  return {
    results,
    loading,
    granularity,
    load,
    unload,
    isLoaded,
    reloadLoadedResults,
    getTestsResultsAt,
    getResultsByRun,
  };
});
