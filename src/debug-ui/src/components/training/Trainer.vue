<template>
    <div class="row">
        <div class="col-auto">
            <span class="badge text-bg-info m-1"> {{ algorithm }} </span>
            <span class="badge text-bg-success m-1"> {{ level }} </span>
            <span v-for="wrapper in wrappers" class="badge text-bg-warning m-1"> {{ wrapper }} </span>

            <div class="row">
                <div class="mx-auto col-auto">
                    <div class="input-group">
                        <span class="input-group-text"> Train for </span>
                        <input type="number" class="form-control" v-model.number="trainSteps" @keyup.enter="train"
                            size="2" />
                        <span class="input-group-text"> steps </span>
                        <button type="button" class="btn btn-success" @click="train">
                            Train
                            <font-awesome-icon icon="fa-solid fa-solid fa-forward-step" />
                        </button>
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-auto mx-auto">
                    <table class="table table-sm">
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
                            <tr v-if="isTraining">
                                <font-awesome-icon icon="fa-solid fa-solid fa-spinner" spin />
                            </tr>
                            <tr v-for="(episode, i) in episodes" @click="() => onEpisodeSelected(episodes[i])"
                                class="table-item">
                                <td> Episode {{ episode }}</td>
                                <td> {{ episodeSteps[i] }} </td>
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
                <EpisodeViewer v-if="selectedEpisode != null" class="col-auto" :frames="frames"
                    :episode="selectedEpisode" />
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { WS_URL } from '../../constants';
import { ReplayEpisode } from '../../models/Episode';
import { Metrics } from '../../models/Metric';
import { useEpisodeStore } from '../../stores/EpisodeStore';
import MetricsPlotter from '../charts/MetricsPlotter.vue';
import EpisodeViewer from '../replay/EpisodeViewer.vue';


const replayStore = useEpisodeStore();
const episodes = ref([] as number[]);
const episodeSteps = ref([] as number[]);
const metrics = ref([] as Metrics[]);
const trainSteps = ref(20);
const isTraining = ref(false);
const selectedEpisode = ref(null as ReplayEpisode | null);
const frames = ref([] as string[]);
defineProps<{
    algorithm: string,
    level: string,
    wrappers: string[]
}>();


function train() {
    const ws = new WebSocket(WS_URL);
    ws.onmessage = (event: MessageEvent) => {
        const data = JSON.parse(event.data) as { step: number, episode: number, metrics: Metrics };
        episodes.value.unshift(data.episode);
        episodeSteps.value.unshift(data.step);
        metrics.value.unshift(data.metrics);
    }
    ws.onopen = () => {
        isTraining.value = true;
        ws.send(JSON.stringify({ steps: trainSteps.value }));
    }
    ws.onclose = () => {
        isTraining.value = false;
    }
}



function onEpisodeSelected(episodeNum: number) {
    console.log("Episode selected: ", episodeNum);
    replayStore.getCurrentTrainEpisode(episodeNum).then(episode => selectedEpisode.value = episode);

    replayStore.getCurrentTrainFrames(episodeNum).then(f => {
        frames.value = f;
    });
}

</script>

<style scoped>
.table-item:hover {
    cursor: pointer;
    background-color: #e9ecef;
}
</style>