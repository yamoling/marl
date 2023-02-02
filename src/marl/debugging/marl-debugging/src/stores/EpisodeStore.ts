import { ref } from "vue";
import { defineStore } from "pinia";
import { Episode } from "../models/Episode";
import { Metrics } from "../models/Metric";

export const useEpisodeStore = defineStore("ReplayStore", () => {


    const trainingList = ref([] as string[]);
    const testingList = ref([] as string[]);
    const trainMetrics = ref([] as Metrics[]);
    const testMetrics = ref([] as Metrics[]);


    function refresh() {
        fetch("http://0.0.0.0:5174/list/train")
            .then(resp => resp.json())
            .then(trainList => trainingList.value = trainList.map((e: string) => e.substring(0, e.length - 5)));

        fetch("http://0.0.0.0:5174/list/test")
            .then(resp => resp.json())
            .then(testList => testingList.value = testList);

        fetch("http://0.0.0.0:5174/metrics/train")
            .then(resp => resp.json())
            .then(metrics => trainMetrics.value = metrics);

        fetch("http://0.0.0.0:5174/metrics/test")
            .then(resp => resp.json())
            .then(metrics => testMetrics.value = metrics);
    }
    refresh();


    async function getEpisode(kind: "test" | "train", num: number): Promise<Episode> {
        const resp = await fetch("http://0.0.0.0:5174/episode/" + kind + "/" + num);
        return await resp.json();
    }


    return { trainMetrics, testMetrics, trainingList, testingList, getEpisode, refresh };
});
