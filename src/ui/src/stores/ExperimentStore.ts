import { ref } from "vue";
import { defineStore } from "pinia";
import { HTTP_URL } from "../constants";
import { ExperimentInfo } from "../models/Infos";
import { Experiment } from "../models/Experiment";

export const useExperimentStore = defineStore("ExperimentStore", () => {

    const experimentInfos = ref(new Map() as Map<string, ExperimentInfo>);
    const loading = ref(false);

    function refresh() {
        loading.value = true;
        fetch(`${HTTP_URL}/experiment/list`)
            .then(resp => resp.json())
            .then((infos: object) => {
                for (const [key, value] of Object.entries(infos)) {
                    const experiment = value;
                    experiment.train = ref(value.train);
                    experiment.test = ref(value.test);
                    experimentInfos.value.set(key, experiment);
                }
                loading.value = false;
            });
    }
    refresh();

    async function deleteExperiment(logdir: string) {
        await deleteExperiments([logdir]);
    }

    async function deleteExperiments(logdirs: string[]) {
        loading.value = true;
        await Promise.all(logdirs.map(async logdir => {
            try {
                await fetch(`${HTTP_URL}/experiment/delete/${logdir}`, { method: "DELETE" });
                experimentInfos.value.delete(logdir);
            } catch (e: any) {
                alert(e.message);
            }
        }));
        loading.value = false;
    }

    async function loadExperiment(logdir: string): Promise<Experiment> {
        loading.value = true;
        const resp = await fetch(`${HTTP_URL}/experiment/load/${logdir}`);
        if (!resp.ok) {
            loading.value = false;
            throw new Error(await resp.text());
        }
        const experiment = await resp.json() as Experiment;
        loading.value = false;
        return experiment;
    }

    async function unloadExperiment(logdir: string) {
        await fetch(`${HTTP_URL}/experiment/load/${logdir}`, { method: "DELETE" });
    }

    async function createExperiment(logdir: string, params: any) {
        const data = JSON.stringify(params);
        const url = `${HTTP_URL}/experiment/create`;
        // 1 crete the experiment
        const resp = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: data
        });
        // 2 Add the experimentInfo to the store
        experimentInfos.value.set(logdir, await resp.json());
        // 3 refhresh the experiment info table
        refresh();
        return logdir;
    }

    return {
        experimentInfos,
        loading,
        refresh,
        createExperiment,
        deleteExperiment,
        loadExperiment,
        unloadExperiment,
    };
});
