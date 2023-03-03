import { ref } from "vue";
import { defineStore } from "pinia";
import { ReplayEpisode } from "../models/Episode";
import { Metrics } from "../models/Metric";
import { HTTP_URL } from "../constants";
import { Test } from "../models/Test";

export const useReplayStore = defineStore("ReplayStore", () => {

    const logdirs = ref([] as string[]);
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
        fetch(`${HTTP_URL}/replay/train/list`)
            .then(resp => resp.json())
            .then(trainList => {
                trainingList.value = trainList;
                loadingTrain.value = false;
            });

        fetch(`${HTTP_URL}/replay/test/list`)
            .then(resp => resp.json())
            .then(testList => {
                testingList.value = testList;
                loadingTests.value = false;
                testEpisodeMetrics.value = testingList.value.map(t => new Array(t.episodes.length));
            });

        fetch(`${HTTP_URL}/ls/logs`)
            .then(resp => resp.json())
            .then(d => {
                const dirs = d as any[];
                logdirs.value = dirs.map(d => d.path);
            });

    }
    refresh();


    async function loadTestEpisodeMetrics(stepNum: number, episodeNum: number) {
        if (testEpisodeMetrics.value[stepNum][episodeNum] == undefined) {
            const resp = await fetch(`${HTTP_URL}/metrics/test/${stepNum}/${episodeNum}`);
            testEpisodeMetrics.value[stepNum][episodeNum] = await resp.json();
        }
    }

    async function getEpisode(directory: string): Promise<ReplayEpisode> {
        const resp = await fetch(`${HTTP_URL}/replay/episode/${directory}`);
        return await resp.json();
    }

    async function getTestEpisode(step: number, index: number): Promise<ReplayEpisode> {
        const resp = await fetch(`${HTTP_URL}/episode/test/${step}/${index}`);
        return await resp.json();
    }

    async function getTestFrames(step: number, index: number): Promise<string[]> {
        const resp = await fetch(`${HTTP_URL}/frames/test/${step}/${index}`);
        return await resp.json();
    }

    async function getTrainFrames(episodeNum: number): Promise<string[]> {
        const resp = await fetch(`${HTTP_URL}/frames/train/${episodeNum}`);
        return await resp.json();
    }

    async function getCurrentTrainEpisode(episodeNum: number): Promise<ReplayEpisode> {
        const resp = await fetch(`${HTTP_URL}/train/episode/${episodeNum}`);
        return await resp.json();
    }

    async function getCurrentTrainFrames(episodeNum: number): Promise<string[]> {
        const resp = await fetch(`${HTTP_URL}/train/frames/${episodeNum}`);
        return await resp.json();
    }


    return { logdirs, loadingMetrics, loadingTests, loadingTrain, testMetrics, trainingList, testingList, testEpisodeMetrics, getEpisode, getCurrentTrainFrames, getCurrentTrainEpisode, getTestEpisode, refresh, getTestFrames, loadTestEpisodeMetrics, getTrainFrames };
});
