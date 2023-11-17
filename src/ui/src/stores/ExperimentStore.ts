import { defineStore } from "pinia";
import { HTTP_URL } from "../constants";
import { Experiment } from "../models/Experiment";
import { ReplayEpisodeSummary } from "../models/Episode";

export const useExperimentStore = defineStore("ExperimentStore", () => {


    async function getAllExperiments(): Promise<Experiment[]> {
        try {
            const resp = await fetch(`${HTTP_URL}/experiment/list`);
            if (!resp.ok) {
                throw new Error("Failed to load experiments: " + await resp.text());
            }
            return await resp.json() as Experiment[];
        } catch (e: any) {
            throw new Error("Failed to load experiments: " + e.message);
        }
    }

    async function getExperiment(logdir: string): Promise<Experiment | null> {
        try {
            const resp = await fetch(`${HTTP_URL}/experiment/${logdir}`);
            if (!resp.ok) {
                throw new Error("Failed to load experiment: " + await resp.text());
            }
            return await resp.json();
        } catch (e: any) {
            throw new Error("Failed to load experiment: " + e.message);
        }
    }

    async function loadExperiment(logdir: string) {
        return await fetch(`${HTTP_URL}/experiment/load/${logdir}`, { method: "POST" })
    }

    async function unloadExperiment(logdir: string) {
        return await fetch(`${HTTP_URL}/experiment/load/${logdir}`, { method: "DELETE" })
    }


    async function getTestEpisodes(logdir: string, time_step: number): Promise<ReplayEpisodeSummary[]> {
        const resp = await fetch(`${HTTP_URL}/experiment/test/list/${time_step}/${logdir}`);
        return await resp.json();
    }

    return {
        getAllExperiments,
        getExperiment,
        loadExperiment,
        unloadExperiment,
        getTestEpisodes,
    };
});
