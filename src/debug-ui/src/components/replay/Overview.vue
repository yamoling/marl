<template>
    <div class="row">
        <div class="col-auto">
            <button class="btn btn-primary" @click="globalState.refreshExperiment">Refresh</button>
        </div>
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
                    <tr v-for="train in globalState.experiment?.train" @click="() => onEpisodeClicked(train)">
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
                    <th class="px-1"> # Step </th>
                    <th class="px-1"> Length</th>
                    <th class="px-1"> Score </th>
                    <th class="px-1"> Gems </th>
                    <th class="px-1"> Elevator</th>
                </thead>
                <tbody>
                    <tr v-for="test in globalState.experiment?.test" @click="() => onTestClicked(test)"
                        :class="(test.name == selectedTestEpisode?.name) ? 'selected' : ''">
                        <td> {{ test.name }} </td>
                        <td> {{ test.metrics.avg_episode_length }}</td>
                        <td> {{ test.metrics.avg_score }} </td>
                        <td> {{ test.metrics.avg_gems_collected }}</td>
                        <td> {{ test.metrics.avg_in_elevator }}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        <div class="col-auto mx-auto text-center" v-if="selectedTestEpisode != null">
            <h4>Tests at time step {{ selectedTestEpisode.name }}</h4>
            <table class="table table-sm table-striped table-hover">
                <thead>
                    <tr>
                        <th> # Test </th>
                        <th> Length </th>
                        <th> Score </th>
                        <th> Gems </th>
                        <th> Elevator </th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-if="testsAtStep.length == 0">
                        <td colspan="5">
                            <font-awesome-icon icon="spinner" spin />
                        </td>
                    </tr>
                    <tr v-for="test in testsAtStep" @click="() => onEpisodeClicked(test)">
                        <td> {{ test.name }} </td>
                        <td> {{ test.metrics.episode_length }} </td>
                        <td> {{ test.metrics.score }} </td>
                        <td> {{ test.metrics.gems_collected }} </td>
                        <td> {{ test.metrics.in_elevator }} </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</template>
<script setup lang="ts">
import { ref } from 'vue';
import { useGlobalState } from '../../stores/GlobalState';
import type { ReplayEpisodeSummary } from "../../models/Episode";
import { useReplayStore } from '../../stores/ReplayStore';


const testList = ref({} as HTMLElement);
const globalState = useGlobalState();
const replayStore = useReplayStore();
const selectedTestEpisode = ref(null as ReplayEpisodeSummary | null);
const testsAtStep = ref([] as ReplayEpisodeSummary[]);


async function onTestClicked(test: ReplayEpisodeSummary) {
    selectedTestEpisode.value = test;
    testsAtStep.value = [];
    testsAtStep.value = await replayStore.getTestEpisodes(test.directory)
}

async function onEpisodeClicked(episode: ReplayEpisodeSummary) {
    emits("requestViewEpisode", replayStore.getEpisode(episode.directory));
}

const emits = defineEmits(["requestViewEpisode"]);

</script>
<style>
tr.selected {
    background-color: gainsboro;
}
</style>