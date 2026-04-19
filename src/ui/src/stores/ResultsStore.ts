import { defineStore } from "pinia";
import { Dataset, ExperimentResults } from "../models/Experiment";
import { HTTP_URL } from "../constants";
import { ReplayEpisodeSummary } from "../models/Episode";
import { ref, watch } from "vue";

export const useResultsStore = defineStore("ResultsStore", () => {

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

    async function load(logdir: string, defaultGranularity?: number): Promise<ExperimentResults> {
        const activeGranularity = currentGranularity(defaultGranularity);
        loading.value.set(logdir, true);
        try {
            const resp = await fetch(`${HTTP_URL}/results/load/${logdir}?granularity=${activeGranularity}`);
            const datasets = await resp.json() as Dataset[];
            const experimentResults = new ExperimentResults(logdir, datasets);
            results.value.set(logdir, experimentResults);
            return experimentResults;
        } finally {
            loading.value.set(logdir, false);
        }
    }

    function unload(logdir: string) {
        results.value.delete(logdir);
    }

    async function reloadLoadedResults() {
        if (granularity.value == null) {
            return;
        }
        await Promise.all(Array.from(results.value.keys()).map((logdir) => load(logdir, granularity.value ?? undefined)));
    }

    /**
     * Get the unagregated test results for a given experiment at a given time step.
     */
    async function getTestsResultsAt(logdir: string, timeStep: number): Promise<ReplayEpisodeSummary[]> {
        const resp = await fetch(`${HTTP_URL}/results/test/${timeStep}/${logdir}`);
        return await resp.json();
    }

    async function getResultsByRun(logdir: string): Promise<ExperimentResults[]> {
        const activeGranularity = granularity.value;
        const granularityQuery = activeGranularity == null ? "" : `?granularity=${activeGranularity}`;
        const resp = await fetch(`${HTTP_URL}/results/load-by-run/${logdir}${granularityQuery}`);
        const datasets = await resp.json() as Dataset[][];
        return datasets.filter(ds => ds.length > 0).map(ds => new ExperimentResults(ds[0].logdir, ds));
    }

    function isLoaded(logdir: string): boolean {
        return results.value.has(logdir);
    }

    watch(granularity, async (newGranularity, oldGranularity) => {
        if (newGranularity !== oldGranularity) {
            await reloadLoadedResults();
        }
    })

    return {
        results,
        loading,
        granularity,
        load,
        unload,
        isLoaded,
        reloadLoadedResults,
        getTestsResultsAt,
        getResultsByRun
    };
});
