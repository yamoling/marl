import { defineStore } from "pinia";
import { HTTP_URL } from "../constants";
import { Experiment } from "../models/Experiment";
import { ReplayEpisodeSummary } from "../models/Episode";
import { computed, ref } from "vue";
import { fetchWithJSON } from "../utils";
import { useRunStore } from "./RunStore";

export const useExperimentStore = defineStore("ExperimentStore", () => {
    const loading = ref(false);
    const experiments = ref<Experiment[]>([]);
    const isRunning = computed(() => {
        const res = {} as {[logdir: string]: boolean};
        runStore.runs.forEach((runs, logdir) => {
            res[logdir] = runs.some(run => run.pid != null);
        });
        return res;
    });
    // const runningExperiments = ref(new Set<string>());
    const runStore = useRunStore();
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
                runStore.refresh(exp.logdir);
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
        const resp = await fetchWithJSON(`${HTTP_URL}/experiment/rename`, { logdir, newLogdir });
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
            runStore.remove(logdir);
        }
    }

    async function testOnOtherEnvironment(logdir: string, newLogdir: string, envLogdir: string, nTests: number): Promise<void> {
        await fetchWithJSON(`${HTTP_URL}/experiment/test-on-other-env`, { logdir, newLogdir, envLogdir, nTests });
        refresh()
    }

    async function getEnvImage(logdir: String, seed: number): Promise<string> {
        const resp = await fetch(`${HTTP_URL}/experiment/image/${seed}/${logdir}`);
        return await resp.text();
    }


    async function newRun(logdir: string, nRuns: number, seed: number, nTests: number) {
        const resp = await fetchWithJSON(`${HTTP_URL}/runner/new/${logdir}`, { seed, nTests, nRuns }, "POST");
        if (!resp.ok) {
            alert("Failed to start new run: " + await resp.text());
            return;
        }
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
        rename,
        testOnOtherEnvironment,
        getEnvImage,
    };
});
