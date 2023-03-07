import { ref, watch } from "vue";
import { defineStore } from "pinia";
import { Experiment } from "../models/Experiment";
import { HTTP_URL } from "../constants";
import { ReplayEpisode } from "../models/Episode";

export const useGlobalState = defineStore("GlobalState", () => {
    const logdir = ref(null as string | null);
    const wsPort = ref(null as number | null);
    const experiment = ref(null as Experiment | null);
    const viewingEpisode = ref(null as ReplayEpisode | null);

    watch(logdir, (newValue, oldValue) => {
        experiment.value = null;
        if (newValue != null) {
            getExperiment().then(resp => experiment.value = resp);
        }
    });


    async function getExperiment(): Promise<Experiment> {
        const resp = await fetch(`${HTTP_URL}/load/${logdir.value}`);
        const experiment = await resp.json();
        return experiment;
    }

    async function refreshExperiment() {
        experiment.value = await getExperiment();
    }

    return { logdir, experiment, viewingEpisode, wsPort, refreshExperiment };
});
