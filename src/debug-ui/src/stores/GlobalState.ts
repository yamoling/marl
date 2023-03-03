import { ref, watch } from "vue";
import { defineStore } from "pinia";
import { Experiment } from "../models/Experiment";
import { HTTP_URL } from "../constants";
import { ReplayEpisode } from "../models/Episode";

export const useGlobalState = defineStore("GlobalState", () => {
    const logdir = ref(null as string | null);
    const experiment = ref(null as Experiment | null);
    const viewingEpisode = ref(null as ReplayEpisode | null);

    watch(logdir, () => {
        console.log("changed logdir");
        getExperiment().then(resp => experiment.value = resp);
        return;
    });


    async function getExperiment(): Promise<Experiment> {
        const resp = await fetch(`${HTTP_URL}/load/${logdir.value}`);
        const experiment = await resp.json();
        return experiment;
    }

    return { logdir, experiment, viewingEpisode };
});
