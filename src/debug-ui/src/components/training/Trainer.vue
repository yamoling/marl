<template>
    <div class="row">
        <div class="col-auto">
                <div class="row mb-1">
                    <div class="mx-auto col-auto">
                        <div class="input-group">
                            <span class="input-group-text"> Train for </span>
                            <input type="number" class="form-control" :disabled="isBusy" v-model.number="trainSteps"
                                @keyup.enter="train" size="2" />
                            <span class="input-group-text"> steps </span>
                            <button type="button" class="btn btn-success" @click="train" :disabled="isBusy">
                                Train
                                <font-awesome-icon v-if="!isTraining" icon="fa-solid fa-solid fa-forward-step" />
                                <font-awesome-icon v-else icon="fa-solid fa-solid fa-spinner" spin />
                            </button>
                        </div>
                    </div>
                </div>
                <div class="row mb-1">
                    <div class="mx-auto col-auto">
                        <div class="input-group">
                            <span class="input-group-text"> Test </span>
                            <input type="number" class="form-control" :disabled="isBusy" v-model.number="numTests"
                                @keyup.enter="test" size="2" />
                            <span class="input-group-text"> episodes </span>
                            <button type="button" class="btn btn-secondary" @click="test" :disabled="isBusy">
                                Test
                                <font-awesome-icon v-if="!isTesting" icon="fa-solid fa-solid fa-forward-step" />
                                <font-awesome-icon v-else icon="fa-solid fa-solid fa-spinner" spin />
                            </button>
                        </div>
                    </div>
                </div>

                <div class="row">
                    <div class="col-auto mx-auto">
                        <table class="table table-sm text-center">
                            <thead>
                                <tr>
                                    <th> Episode </th>
                                    <th> Step </th>
                                    <th> Length </th>
                                    <th> Score </th>
                                    <th> Gems </th>
                                    <th> In elevator </th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr v-for="(episode, i) in episodes" @click="() => onEpisodeSelected(episodes[i])"
                                    class="table-item" :class="episode == selectedEpisodeNum ? 'selected' : ''">
                                    <td> Episode {{ episode }}</td>
                                    <td> {{ steps[i] }} </td>
                                    <td> {{ metrics[i].episode_length }}</td>
                                    <td> {{ metrics[i].score }} </td>
                                    <td> {{ metrics[i].gems_collected }} </td>
                                    <td> {{ metrics[i].in_elevator }} </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            <div class="col-5">
                <div class="row">
                    <MetricsPlotter :metrics="metrics" :reverse-labels="true" :max-steps="50" />
                </div>
                <div class="row">
                    <EpisodeViewer ref="episodeViewer" v-if="selectedEpisode != null" class="col-auto" :frames="frames"
                        :episode="selectedEpisode" @close="() => selectedEpisode = null" />
            </div>
            <div class="row">
                <ReplayMemory />
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { WS_URL } from '../../constants';
import { ReplayEpisode } from '../../models/Episode';
import { Metrics } from '../../models/Metric';
import { useReplayStore } from '../../stores/ReplayStore';
import MetricsPlotter from '../charts/MetricsPlotter.vue';
import EpisodeViewer from '../replay/EpisodeViewer.vue';
import ReplayMemory from '../visualisation/ReplayMemory.vue';


const replayStore = useReplayStore();
const episodes = ref([] as number[]);
const steps = ref([] as number[]);
const metrics = ref([] as Metrics[]);
const trainSteps = ref(20);
const numTests = ref(10);
const isTraining = ref(false);
const isTesting = ref(false);
const selectedEpisode = ref(null as ReplayEpisode | null);
const selectedEpisodeNum = ref(null as number | null);
const frames = ref([] as string[]);
const episodeViewer = ref(null as typeof EpisodeViewer | null);
const isBusy = computed(() => isTesting.value || isTraining.value);


function train() {
    const ws = new WebSocket(WS_URL);
    ws.onmessage = (event: MessageEvent) => {
        const data = JSON.parse(event.data) as { step: number, episode: number, metrics: Metrics };
        episodes.value.unshift(data.episode);
        steps.value.unshift(data.step);
        metrics.value.unshift(data.metrics);
    }
    ws.onopen = () => {
        isTraining.value = true;
        ws.send(JSON.stringify({ type: "train", steps: trainSteps.value }));
    }
    ws.onclose = () => isTraining.value = false;
}

function test() {
    const ws = new WebSocket(WS_URL);
    ws.onmessage = (event: MessageEvent) => {
        const data = JSON.parse(event.data) as { step: number, episode: number, metrics: Metrics };
        episodes.value.unshift(data.episode);
        steps.value.unshift(data.step);
        metrics.value.unshift(data.metrics);
    }
    ws.onopen = () => {
        isTesting.value = true;
        ws.send(JSON.stringify({ type: "test", numTests: numTests.value }));
    }
    ws.onclose = () => isTesting.value = false;
}



function onEpisodeSelected(episodeNum: number) {
    console.log("Episode selected: ", episodeNum);
    selectedEpisodeNum.value = episodeNum;
    replayStore.getCurrentTrainEpisode(episodeNum).then(episode => selectedEpisode.value = episode);
    replayStore.getCurrentTrainFrames(episodeNum).then(f => frames.value = f);
}

function reset() {
    episodeViewer.value?.reset();
    episodes.value = [];
    steps.value = [];
    metrics.value = [];
    selectedEpisode.value = null;
    frames.value = [];
    isTraining.value = false;
    isTesting.value = false;
}

defineExpose({ reset });

</script>

<style scoped>
.table-item:hover {
    cursor: pointer;
    background-color: #e9ecef;
}

.table-item.selected {
    background-color: #e9ecef;
}
</style>