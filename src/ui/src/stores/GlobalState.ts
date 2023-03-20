import { ref, watch } from "vue";
import { defineStore } from "pinia";
import { Experiment } from "../models/Experiment";
import { HTTP_URL } from "../constants";

export const useGlobalState = defineStore("GlobalState", () => {
    const logdir = ref(null as string | null);
    const wsPort = ref(null as number | null);
    const experiment = ref(null as Experiment | null);
    const loading = ref(false);

    watch(logdir, (newValue, oldValue) => {
        experiment.value = null;
        if (newValue != null) {
            refreshExperiment();
        }
    });

    async function getExperiment(): Promise<Experiment> {
        try {
            loading.value = true;
            const resp = await fetch(`${HTTP_URL}/load/${logdir.value}`);
            const res = await resp.json();
            loading.value = false;
            return res;
        } catch {
            throw new Error(`Error while loading the experiment from ${logdir.value}`);
        } finally {
            loading.value = false;
        }
    }

    async function refreshExperiment() {
        try {
            experiment.value = await getExperiment();
        } catch (e) {
            alert((e as Error).message);
        }
    }

    async function loadCheckpoint(directory: string): Promise<void> {
        await fetch(`${HTTP_URL}/replay/checkpoint/load/${directory}`);
    }

    return { logdir, experiment, wsPort, refreshExperiment, loading, loadCheckpoint };
});
