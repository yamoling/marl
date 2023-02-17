import { ref } from "vue";
import { defineStore } from "pinia";
import { Episode } from "../models/Episode";
import { Metrics } from "../models/Metric";
import { HTTP_URL } from "../constants";
import { Test } from "../models/Test";

export const useEpisodeStore = defineStore("ReplayStore", () => {

    const loadingTests = ref(false);
    const loadingTrain = ref(false);
    const loadingMetrics = ref(false);
    const trainingList = ref([] as string[]);
    const testingList = ref([] as Test[]);
    const testMetrics = ref([] as Metrics[]);
    const testEpisodeMetrics = ref([] as Metrics[][]);

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
                testEpisodeMetrics.value = testingList.value.map(t => new Array(t.episodes.length));
            });

        fetch(`${HTTP_URL}/metrics/test`)
            .then(resp => resp.json())
            .then(metrics => {
                testMetrics.value = metrics;
                loadingMetrics.value = false;
            });
    }
    refresh();


    async function loadTestEpisodeMetrics(stepNum: number, episodeNum: number) {
        console.log(stepNum, episodeNum, testEpisodeMetrics.value[stepNum][episodeNum]);
        if (testEpisodeMetrics.value[stepNum][episodeNum] == undefined) {
            const resp = await fetch(`${HTTP_URL}/metrics/test/${stepNum}/${episodeNum}`);
            testEpisodeMetrics.value[stepNum][episodeNum] = await resp.json();
        }
    }

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


    return { loadingMetrics, loadingTests, loadingTrain, testMetrics, trainingList, testingList, testEpisodeMetrics, getTrainEpisode, getTestEpisode, refresh, getTestFrames, loadTestEpisodeMetrics };
});
