import { defineStore } from "pinia";
import { HTTP_URL } from "../constants";
import { Experiment, ExperimentSchema } from "../models/Experiment";
import { ReplayEpisodeSummary } from "../models/Episode";
import { computed, ref } from "vue";
import { fetchWithJSON } from "../utils";
import { useRunStore } from "./RunStore";

export const useExperimentStore = defineStore("ExperimentStore", () => {
    const loading = ref(false);
    const experiments = ref([] as Experiment[]);
    const runStore = useRunStore();
    const isRunning = computed(() => {
        const res = {} as { [logdir: string]: boolean };
        runStore.runs.forEach((runs, logdir) => {
            res[logdir] = runs.some(run => run.pid != null);
        });
        return res;
    });
    refresh()

    async function loadExperiments() {
        try {
            loading.value = true;
            const resp = await fetch(`${HTTP_URL}/experiment/list`, { headers: { "Access-Control-Allow-Origin": "*" } });
            const json = await resp.json();
            const experiments = ExperimentSchema.array().parse(json);
            for (const exp of experiments) {
                runStore.refresh(exp.logdir);
            }
            return experiments
        } catch (e: any) {
            alert("Failed to load experiments: " + e.message);
            return [];
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
            return ExperimentSchema.parse(await resp.json());
        } catch (e: any) {
            alert(e.message);
            return null;
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
        experiments.value = await loadExperiments();
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

    async function stopRuns(logdir: string) {
        const resp = await fetch(`${HTTP_URL}/experiment/stop-runs/${logdir}`, { method: "POST" });
        if (!resp.ok) {
            alert("Failed to stop runs: " + await resp.text());
            return;
        }
        await runStore.refresh(logdir);
    }


    async function getEnvImage(logdir: String, seed: number): Promise<string> {
        const resp = await fetch(`${HTTP_URL}/experiment/image/${seed}/${logdir}`);
        return await resp.text();
    }


    async function newRun(
        logdir: string,
        nRuns: number,
        seed: number,
        nTests: number,
        gpuStrategy: "scatter" | "group" = "scatter",
        disabledDevices: number[] = [],
    ) {
        const resp = await fetchWithJSON(
            `${HTTP_URL}/runner/new/${logdir}`,
            { seed, nTests, nRuns, gpuStrategy, disabledDevices },
            "POST",
        );
        if (!resp.ok) {
            alert("Failed to start new run: " + await resp.text());
            return false;
        }
        await runStore.refresh(logdir);
        return true;
    }

    async function refresh() {
        experiments.value = await loadExperiments();
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
        stopRuns,
        rename,
        getEnvImage,
        newRun,
    };
});
