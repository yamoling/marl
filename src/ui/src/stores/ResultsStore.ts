import { defineStore } from "pinia";
import { ExperimentResults } from "../models/Experiment";
import { HTTP_URL } from "../constants";
import { stringToRGB } from "../utils";
import { ReplayEpisodeSummary } from "../models/Episode";

export const useResultsStore = defineStore("ResultsStore", () => {


    async function loadExperimentResults(logdir: string): Promise<ExperimentResults> {
        const resp = await fetch(`${HTTP_URL}/results/load/${logdir}`);
        const results = await resp.json() as ExperimentResults;
        const colours = JSON.parse(localStorage.getItem("datasetColours") || "{}") as any;
        results.colour = colours[logdir] || stringToRGB(logdir);
        results.train.forEach(ds => ds.colour = colours[logdir] || stringToRGB(logdir));
        results.test.forEach(ds => ds.colour = colours[logdir] || stringToRGB(logdir));
        return results;
    }

    /**
     * Get the unagregated test results for a given experiment at a given time step.
     */
    async function getTestsResultsAt(logdir: string, timeStep: number): Promise<ReplayEpisodeSummary[]> {
        const resp = await fetch(`${HTTP_URL}/results/test/${timeStep}/${logdir}`);
        return await resp.json();
    }

    return {
        loadExperimentResults,
        getTestsResultsAt
    };
});
