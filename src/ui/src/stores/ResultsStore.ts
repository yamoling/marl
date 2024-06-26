import { defineStore } from "pinia";
import { Dataset, ExperimentResults } from "../models/Experiment";
import { HTTP_URL } from "../constants";
import { ReplayEpisodeSummary } from "../models/Episode";
import { ref } from "vue";

export const useResultsStore = defineStore("ResultsStore", () => {

    const results = ref(new Map<string, ExperimentResults>());
    const loading = ref(new Map<string, boolean>());


    async function load(logdir: string): Promise<ExperimentResults> {
        loading.value.set(logdir, true);
        const resp = await fetch(`${HTTP_URL}/results/load/${logdir}`);
        const response = await resp.json() as Dataset[];
        const experimentResults = new ExperimentResults(logdir, response);
        results.value.set(logdir, experimentResults);
        loading.value.set(logdir, false);
        return experimentResults;
    }

    function unload(logdir: string) {
        results.value.delete(logdir);
    }

    /**
     * Get the unagregated test results for a given experiment at a given time step.
     */
    async function getTestsResultsAt(logdir: string, timeStep: number): Promise<ReplayEpisodeSummary[]> {
        const resp = await fetch(`${HTTP_URL}/results/test/${timeStep}/${logdir}`);
        return await resp.json();
    }

    async function getResultsByRun(logdir: string): Promise<ExperimentResults[]> {
        const resp = await fetch(`${HTTP_URL}/results/load-by-run/${logdir}`);
        const results = await resp.json() as ExperimentResults[];
        return results;
    }

    function isLoaded(logdir: string): boolean {
        return results.value.has(logdir);
    }

    return {
        results,
        loading,
        load,
        unload,
        isLoaded,
        getTestsResultsAt,
        getResultsByRun
    };
});
