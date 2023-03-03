<template>
    <div class="row">
        <div class="col-auto text-center">
            <h4> Training </h4>
            <table class="table table-sm table-striped table-hover text-center">
                <thead style="position: sticky; top: 0; z-index: 1;">
                    <th class="px-1"> # Episode </th>
                    <th class="px-1"> Length</th>
                    <th class="px-1"> Score </th>
                    <th class="px-1"> Gems </th>
                    <th class="px-1"> Elevator</th>
                </thead>
                <tbody>
                    <tr v-for="train in globalState.experiment?.train" @click="() => onTrainEpisodeClicked(train)">
                        <td> {{ train.name }} </td>
                        <td> {{ train.metrics.episode_length }}</td>
                        <td> {{ train.metrics.score }} </td>
                        <td> {{ train.metrics.gems_collected }}</td>
                        <td> {{ train.metrics.in_elevator }}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        <div class="col-auto text-center" ref="testList">
            <h4> Tests </h4>
            <table class="table table-sm table-striped table-hover text-center">
                <thead style="position: sticky; top: 0; z-index: 1;">
                    <th class="px-1"> # Episode </th>
                    <th class="px-1"> Length</th>
                    <th class="px-1"> Score </th>
                    <th class="px-1"> Gems </th>
                    <th class="px-1"> Elevator</th>
                </thead>
                <tbody>
                    <tr v-for="test in globalState.experiment?.test" @click="() => onTestClicked(test)">
                        <td> {{ test.name }} </td>
                        <td> {{ test.metrics.episode_length }}</td>
                        <td> {{ test.metrics.score }} </td>
                        <td> {{ test.metrics.gems_collected }}</td>
                        <td> {{ test.metrics.in_elevator }}</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</template>
<script setup lang="ts">
import { ref } from 'vue';
import { Collapse } from "bootstrap/dist/js/bootstrap.bundle.min.js";
import { useGlobalState } from '../stores/GlobalState';
import type { ReplayEpisode } from "../models/Episode";
import { useReplayStore } from '../stores/ReplayStore';


interface Collapsable {
    collapse: () => void,
    toggle: () => void,
    show: () => void
}


const testList = ref({} as HTMLElement);
const globalState = useGlobalState();
const replayStore = useReplayStore();


function onTestClicked(test: ReplayEpisode) {
    console.log(test.directory);
    // replayStore.getTestEpisode()
    // emits("testEpisodeSelected", testNum, episodeNum);
}

async function onTrainEpisodeClicked(episode: ReplayEpisode) {
    globalState.viewingEpisode = await replayStore.getEpisode(episode.directory);
    emits("requestViewEpisode");
}

const emits = defineEmits(["requestViewEpisode"]);

</script>