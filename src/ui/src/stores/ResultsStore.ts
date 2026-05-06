import { defineStore } from "pinia";
import { DatasetSchema, ExperimentResults } from "../models/Results";
import { ReplayEpisodeSummarySchema } from "../models/Episode";
import { ref, watch } from "vue";
import { apiFetch, parseOrThrow } from "../api";
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
      const fallbackGranularity = defaultGranularity ?? settingsStore.settings.granularity;
      granularity.value = normalizeGranularity(fallbackGranularity);
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
      const datasets = parseOrThrow(DatasetSchema.array(), await resp.json());
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
    return parseOrThrow(ReplayEpisodeSummarySchema.array(), json);

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
  };
});
