import { ref } from "vue";
import { defineStore } from "pinia";
import { Episode } from "../models/Episode";
import { Metrics } from "../models/Metric";
import { Test } from "../models/Test";
import { HTTP_URL } from "../constants";

export const useEpisodeStore = defineStore("ReplayStore", () => {

    const loadingTests = ref(false);
    const loadingTrain = ref(false);
    const loadingMetrics = ref(false);
    const trainingList = ref([] as string[]);
    const testingList = ref([] as string[]);
    const testMetrics = ref([] as Metrics[]);


    function refresh() {
        loadingTests.value = true;
        loadingTrain.value = true;
        loadingMetrics.value = true;
        fetch(`${HTTP_URL}/list/train`)
            .then(resp => resp.json())
            .then(trainList => {
                trainingList.value = trainList.map((e: string) => e.substring(0, e.length - 5));
                loadingTrain.value = false;
            });

        fetch(`${HTTP_URL}/list/test`)
            .then(resp => resp.json())
            .then(testList => {
                testingList.value = testList;
                loadingTests.value = false;
            });

        // fetch(`${HTTP_URL}/metrics/train`)
        //     .then(resp => resp.json())
        //     .then(metrics => trainMetrics.value = metrics);

        fetch(`${HTTP_URL}/metrics/test`)
            .then(resp => resp.json())
            .then(metrics => {
                testMetrics.value = metrics;
                loadingMetrics.value = false;
            });
    }
    refresh();


    async function getTrainEpisode(num: number): Promise<Episode> {
        const resp = await fetch(`${HTTP_URL}/episode/train/${num}`);
        return await resp.json();
    }

    async function getTestEpisode(step: number, index: number): Promise<Episode> {
        const resp = await fetch(`${HTTP_URL}/episode/test/${step}/${index}`);
        return await resp.json();
    }

    async function getTestFrames(step: number, index: number): Promise<string[]> {
        const resp = await fetch(`${HTTP_URL}/frames/test/${step}/${index}`);
        return await resp.json();
    }


    return { loadingMetrics, loadingTests, loadingTrain, testMetrics, trainingList, testingList, getTrainEpisode, getTestEpisode, refresh, getTestFrames };
});
