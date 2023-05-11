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
            const infos = await resp.json();
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
        const experimentColour = stringToRGB(experiment.logdir);
        experiment.test_metrics.datasets.forEach(dataset => {
            dataset.colour = experimentColour;
        });
        experiment.train_metrics.datasets.forEach(dataset => {
            dataset.colour = experimentColour;
        });
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

    async function createExperiment(logdir: string, params: any) {
        const data = JSON.stringify(params);
        const url = `${HTTP_URL}/experiment/create`;
        // 1 create the experiment
        const resp = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: data
        });
        // 2 Add the experimentInfo to the store
        experimentInfos.value.push(await resp.json());
        // 3 refhresh the experiment info table
        refresh();
        return logdir;
    }

    async function getTestEpisodes(logdir: string, time_step: number): Promise<ReplayEpisodeSummary[]> {
        const resp = await fetch(`${HTTP_URL}/experiment/test/list/${time_step}/${logdir}`);
        return await resp.json();
    }

    return {
        experimentInfos,
        experiments,
        loading,
        anyLoading,
        refresh,
        createExperiment,
        deleteExperiment,
        loadExperiment,
        unloadExperiment,
        getTestEpisodes
    };
});
