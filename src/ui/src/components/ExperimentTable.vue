<template>
    <div class="row text-center">
        <!-- <div class="col-auto">
            <button class="btn btn-primary" @click="globalState.refreshExperiment">Refresh</button>
        </div> -->
        <div v-show="toShow == 'train'" class="col">
            <h4> Training </h4>
            <div class="table-scrollable">
                <table class="table table-sm table-striped table-hover">
                    <thead>
                        <tr>
                            <th class="px-1"> # Episode </th>
                            <th class="px-1"> Length</th>
                            <th class="px-1"> Score </th>
                            <th class="px-1"> Gems </th>
                            <th class="px-1"> Elevator</th>
                        </tr>
                    </thead>
                    <tbody @scrollend="onTrainScroll">
                        <tr v-for="train in trainList" @click="() => emits('view-episode', train.directory)">
                            <td> {{ train.name }} </td>
                            <td> {{ train.metrics.episode_length }}</td>
                            <td> {{ train.metrics.score }} </td>
                            <td> {{ train.metrics.gems_collected }}</td>
                            <td> {{ train.metrics.in_elevator }}</td>
                        </tr>
                        <tr v-if="isTraining">
                            <td colspan="5">
                                <font-awesome-icon icon="spinner" spin />
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        <div class="col-auto">
            <h4> Tests </h4>
            <div v-show="toShow == 'test'" class="table-scrollable">
                <table class="table table-sm table-striped table-hover text-center table-scrollable">
                    <thead>
                        <tr>
                            <th class="px-1"> # Step </th>
                            <th class="px-1"> Length</th>
                            <th class="px-1"> Score </th>
                            <th class="px-1"> Gems </th>
                            <th class="px-1"> Elevator</th>
                            <th> </th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-for="test in experiment.test" @click="() => onTestClicked(test)"
                            :class="(test.name == selectedTestEpisode?.name) ? 'selected' : ''">
                            <td> {{ test.name }} </td>
                            <td> {{ test.metrics.avg_episode_length }}</td>
                            <td> {{ test.metrics.avg_score }} </td>
                            <td> {{ test.metrics.avg_gems_collected }}</td>
                            <td> {{ test.metrics.avg_in_elevator }}</td>
                            <td @click.stop="() => loadModel(test)" style="cursor: pointer; padding-right: 10px;"
                                title="Load this model">
                                <font-awesome-icon class="text-warning" icon="bolt" />
                            </td>
                        </tr>
                        <tr v-if="isTraining">
                            <td colspan="6">
                                <font-awesome-icon icon="spinner" spin />
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        <div class="col-auto mx-auto text-center" v-if="toShow == 'test' && selectedTestEpisode != null">
            <h4>Tests at time step {{ selectedTestEpisode.name }}</h4>
            <div class="table-scrollable">
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
                        <tr v-for="test in testsAtStep" @click="() => emits('view-episode', test.directory)">
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
    </div>
</template>
<script setup lang="ts">
import { ref, computed } from 'vue';
import type { ReplayEpisodeSummary } from "../models/Episode";
import { useReplayStore } from '../stores/ReplayStore';
import { Experiment } from '../models/Experiment';


const props = defineProps<{
    experiment: Experiment
    isTraining: boolean
    toShow: "test" | "train"
}>();

const replayStore = useReplayStore();
const selectedTestEpisode = ref(null as ReplayEpisodeSummary | null);
const testsAtStep = ref([] as ReplayEpisodeSummary[]);
const trainList = computed(() => {
    // Only take the first 100 items
    return props.experiment.train.slice(trainOffset.value, 100 + trainOffset.value);
});
const trainOffset = ref(0);



async function onTestClicked(test: ReplayEpisodeSummary) {
    selectedTestEpisode.value = test;
    testsAtStep.value = [];
    try {
        testsAtStep.value = await replayStore.getTestEpisodes(test.directory)
    } catch (e) {
        selectedTestEpisode.value = null;
        alert("Failed to load test episodes");
    }
}

async function loadModel(episode: ReplayEpisodeSummary) {
    alert("TODO: Load model")
}


function getScollProgress(e: UIEvent): { downScrollProgress: number, upScrollProgress: number } {
    const target = e.target as HTMLElement;
    const tbodyHeight = target.scrollHeight;
    const downScrollProgress = (target.scrollTop + target.clientHeight) / tbodyHeight;
    const upScrollProgress = target.scrollTop / tbodyHeight;
    return { downScrollProgress, upScrollProgress };
}

function onTrainScroll(e: UIEvent) {
    const { downScrollProgress, upScrollProgress } = getScollProgress(e);
    if (downScrollProgress > 0.9) {
        trainOffset.value = Math.min(trainOffset.value + 10, props.experiment.train.length - 100);
        return;
    }
    if (upScrollProgress < 0.1) {
        trainOffset.value = Math.max(0, trainOffset.value - 10);
        return
    }
}


const emits = defineEmits(["view-episode", "load-model"]);

</script>
