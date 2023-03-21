import { ref } from "vue";
import { defineStore } from "pinia";
import { HTTP_URL } from "../constants";
import { ExperimentInfo } from "../models/Infos";

export const useExperimentStore = defineStore("ExperimentStore", () => {

    const experimentInfos = ref(new Map() as Map<string, ExperimentInfo>);
    const loading = ref(false);

    function refresh() {
        loading.value = true;
        fetch(`${HTTP_URL}/experiment/list`)
            .then(resp => resp.json())
            .then((infos: object) => {
                for (const [key, value] of Object.entries(infos)) {
                    experimentInfos.value.set(key, value);
                }
                loading.value = false;
            });
    }
    refresh();

    async function deleteExperiment(logdir: string) {
        loading.value = true;
        try {
            await fetch(`${HTTP_URL}/experiment/delete/${logdir}`, { method: "DELETE" });
            experimentInfos.value.delete(logdir);
        } catch (e: any) {
            alert(e.message);
        }
        loading.value = false;
    }

    return { experimentInfos, loading, refresh, deleteExperiment };
});
