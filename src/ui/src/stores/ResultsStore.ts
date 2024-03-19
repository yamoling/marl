import { defineStore } from "pinia";
import { ExperimentResults } from "../models/Experiment";
import { HTTP_URL } from "../constants";
import { ReplayEpisodeSummary } from "../models/Episode";
import { ref } from "vue";

export const useResultsStore = defineStore("ResultsStore", () => {

    const results = ref(new Map<string, ExperimentResults>());
    const loading = ref(new Map<string, boolean>());


    async function load(logdir: string): Promise<ExperimentResults> {
        loading.value.set(logdir, true);
        const resp = await fetch(`${HTTP_URL}/results/load/${logdir}`);
        const response = await resp.json() as ExperimentResults;
        response.test.forEach(ds => ds.logdir = logdir);
        response.train.forEach(ds => ds.logdir = logdir);
        console.log(response)
        results.value.set(logdir, response);
        loading.value.set(logdir, false);
        return response;
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
        results.forEach(res => {
            res.test.forEach(ds => ds.logdir = res.logdir);
            res.train.forEach(ds => ds.logdir = res.logdir);
        });
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
