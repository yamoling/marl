import { computed, ref } from "vue";
import { defineStore } from "pinia";
import { HTTP_URL } from "../constants";
import { ExperimentInfo } from "../models/Infos";
import { Experiment } from "../models/Experiment";
import { ReplayEpisodeSummary } from "../models/Episode";
import { stringToRGB } from "../utils";

export const useExperimentStore = defineStore("ExperimentStore", () => {

    const experimentInfos = ref([] as ExperimentInfo[]);
    const experiments = ref([] as Experiment[]);
    const loading = ref([] as boolean[]);
    const anyLoading = computed(() => loading.value.some(l => l));

    async function refresh() {
        loading.value = [true];
        try {
            const resp = await fetch(`${HTTP_URL}/experiment/list`);
            if (!resp.ok) {
                alert(await resp.text());
                return;
            }
            const infos = await resp.json() as ExperimentInfo[];
            experimentInfos.value = infos;
        } catch (e: any) {
            alert(e.message);
        } finally {
            loading.value = experimentInfos.value.map(() => false);
        }
    }
    refresh();

    async function deleteExperiment(logdir: string) {
        await deleteExperiments([logdir]);
    }

    async function deleteExperiments(logdirs: string[]) {
        await Promise.all(logdirs.map(async logdir => {
            const index = experimentInfos.value.findIndex(info => info.logdir === logdir);
            loading.value[index] = true;
            try {
                await fetch(`${HTTP_URL}/experiment/delete/${logdir}`, { method: "DELETE" });
                // Remove the experiment from the list
                experimentInfos.value = experimentInfos.value.filter(info => info.logdir !== logdir);
            } catch (e: any) {
                alert(e.message);
            }
            loading.value[index] = false;
        }));
    }

    async function loadExperiment(logdir: string): Promise<Experiment> {
        const index = experimentInfos.value.findIndex(info => info.logdir === logdir);
        loading.value[index] = true;
        const resp = await fetch(`${HTTP_URL}/experiment/load/${logdir}`);
        if (!resp.ok) {
            loading.value[index] = false;
            throw new Error(await resp.text());
        }
        const experiment = await resp.json() as Experiment;
        experiment.colour = stringToRGB(experiment.logdir);
        // Get experiment index based on the logdir
        const experimentIndex = experiments.value.findIndex(e => e.logdir === logdir);
        if (experimentIndex >= 0) {
            experiments.value[experimentIndex] = experiment;
        } else {
            experiments.value = [experiment, ...experiments.value];
        }
        loading.value[index] = false;
        return experiment;
    }

    async function unloadExperiment(logdir: string) {
        // Remove the experiment from "experiments"
        experiments.value = experiments.value.filter(e => e.logdir !== logdir);
        await fetch(`${HTTP_URL}/experiment/load/${logdir}`, { method: "DELETE" });
    }



    async function getTestEpisodes(logdir: string, time_step: number): Promise<ReplayEpisodeSummary[]> {
        const resp = await fetch(`${HTTP_URL}/experiment/test/list/${time_step}/${logdir}`);
        return await resp.json();
    }

    function isLoaded(logdir: string): boolean {
        return experiments.value.some(e => e.logdir === logdir);
    }

    return {
        experimentInfos,
        experiments,
        loading,
        anyLoading,
        refresh,
        deleteExperiment,
        loadExperiment,
        unloadExperiment,
        getTestEpisodes,
        isLoaded
    };
});
