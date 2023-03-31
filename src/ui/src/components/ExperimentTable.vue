<template>
    <div class="row text-center">
        <div class="col-auto">
            <h4> Tests </h4>
            <div class="table-scrollable">
                <table class="table table-sm table-striped table-hover table-scrollable">
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
                        <tr v-for="[time_step, metrics] in experiment.test_metrics" @click="() => onTestClicked(time_step)"
                            :class="(time_step == selectedTimeStep) ? 'selected' : ''">
                            <td> {{ time_step }} </td>
                            <td> {{ metrics.avg_episode_length?.toFixed(3) }}</td>
                            <td> {{ metrics.avg_score?.toFixed(3) }} </td>
                            <td> {{ metrics.avg_gems_collected?.toFixed(3) }}</td>
                            <td> {{ metrics.avg_in_elevator?.toFixed(3) }}</td>
                            <td @click.stop="loadModel" style="cursor: pointer; padding-right: 10px;"
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
        <div class="col-auto mx-auto text-center" v-if="selectedTimeStep != null">
            <h4>Tests at time step {{ selectedTimeStep }}</h4>
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
                            <td> {{ test.metrics.score.toFixed(3) }} </td>
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
import { ref } from 'vue';
import type { ReplayEpisodeSummary } from "../models/Episode";
import { Experiment } from '../models/Experiment';
import { useExperimentStore } from '../stores/ExperimentStore';


const props = defineProps<{
    experiment: Experiment
    isTraining: boolean
}>();

const store = useExperimentStore();
const selectedTimeStep = ref(null as string | null);
const testsAtStep = ref([] as ReplayEpisodeSummary[]);
// const trainList = computed(() => {
//     // Only take the first 100 items
//     return props.experiment.train.slice(trainOffset.value, 100 + trainOffset.value);
// });



async function onTestClicked(time_step: string) {
    selectedTimeStep.value = time_step;
    testsAtStep.value = [];
    try {
        const tests = await store.getTestEpisodes(props.experiment.logdir, Number.parseInt(time_step));
        console.log(tests)
        testsAtStep.value = tests;
    } catch (e) {
        selectedTimeStep.value = null;
        alert("Failed to load test episodes");
    }
}

async function loadModel() {
    alert("TODO: Load model")
}

/*
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
*/

const emits = defineEmits(["view-episode", "load-model"]);

</script>
