import { defineStore } from "pinia";
import { HTTP_URL } from "../constants";
import { Experiment } from "../models/Experiment";
import { ReplayEpisodeSummary } from "../models/Episode";
import { ref } from "vue";
import { fetchJSON } from "../utils";

export const useExperimentStore = defineStore("ExperimentStore", () => {
    const loading = ref(false);
    const experiments = ref<Experiment[]>([]);
    const runningExperiments = ref(new Set<string>());
    refresh();

    async function refresh() {
        try {
            loading.value = true;
            const resp = await fetch(`${HTTP_URL}/experiment/list`);
            if (!resp.ok) {
                throw new Error("Failed to load experiments: " + await resp.text());
            }
            experiments.value = await resp.json() as Experiment[];
            for (const exp of experiments.value) {
                refreshRunning(exp.logdir).then((running) => {
                    if (running) {
                        runningExperiments.value.add(exp.logdir);
                    } else {
                        runningExperiments.value.delete(exp.logdir);
                    }
                });
            }
        } catch (e: any) {
            throw new Error("Failed to load experiments: " + e.message);
        } finally {
            loading.value = false;
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

    async function refreshRunning(logdir: string): Promise<boolean> {
        try {
            const resp = await fetch(`${HTTP_URL}/experiment/is_running/${logdir}`);
            if (!resp.ok) {
                return false
            }
            return await resp.json();
        } catch (e: any) {
            return false;
        }
    }

    /**
     * Ask the backend to load an experiment, which is required to 
     * replay an episode.
     * @param logdir 
     * @returns 
     */
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

    async function rename(logdir: string, newLogdir: string) {
        const resp = await fetch(`${HTTP_URL}/experiment/rename`, {
            method: "POST",
            body: JSON.stringify({ logdir, newLogdir }),
            headers: { "Content-Type": "application/json" }
        });
        if (!resp.ok) {
            alert("Failed to rename experiment: " + await resp.text());
            return;
        }
        return refresh();
    }

    async function remove(logdir: string) {
        const res = await fetch(`${HTTP_URL}/experiment/delete/${logdir}`, { method: "DELETE" });
        if (!res.ok) {
            alert("Failed to delete experiment: " + await res.text())
        } else {
            experiments.value = experiments.value.filter(exp => exp.logdir !== logdir);
            runningExperiments.value.delete(logdir);
        }
    }

    async function testOnOtherEnvironment(logdir: string, newLogdir: string, envLogdir: string, nTests: number): Promise<void> {
        await fetchJSON(`${HTTP_URL}/experiment/test-on-other-env`, { logdir, newLogdir, envLogdir, nTests });
    }

    async function getEnvImage(logdir: String, seed: number): Promise<string> {
        const resp = await fetch(`${HTTP_URL}/experiment/image/${seed}/${logdir}`);
        return await resp.text();
    }

    return {
        loading,
        experiments,
        runningExperiments,
        refresh,
        getExperiment,
        loadExperiment,
        unloadExperiment,
        getTestEpisodes,
        remove,
        rename,
        testOnOtherEnvironment,
        getEnvImage
    };
});
