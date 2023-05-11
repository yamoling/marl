<template>
    <div v-show="store.experiments.length > 0">
        <h4> Legend </h4>
        <table>
            <tbody>
                <template v-for="exp, in  store.experiments ">
                    <tr class="experiment-row">
                        <td>
                            <input type="color" v-model="exp.test_metrics.datasets[0].colour"
                                @change="(e) => updateColour(exp.logdir, e)">
                        </td>
                        <td> {{ exp.logdir }}</td>
                        <td>
                            <button class="btn btn-sm btn-outline-primary" @click="() => emits('inspect-experiment', exp)">
                                <font-awesome-icon :icon="['fas', 'magnifying-glass']" />
                            </button>
                        </td>
                        <td>
                            <button class="btn btn-sm btn-outline-info" @click="() => toggleExperiment(exp)">
                                <font-awesome-icon v-if="showed.get(exp.logdir)" :icon="['fas', 'eye']" />
                                <font-awesome-icon v-else :icon="['fas', 'eye-slash']" />
                            </button>
                        </td>
                        <td>
                            <button class="btn btn-sm btn-outline-danger" @click="() => store.unloadExperiment(exp.logdir)">
                                <font-awesome-icon :icon="['fas', 'close']" />
                            </button>
                        </td>
                    </tr>
                </template>
            </tbody>
        </table>
    </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue';
import { useExperimentStore } from '../../stores/ExperimentStore';
import { Experiment } from '../../models/Experiment';
import { storeToRefs } from 'pinia';

const store = useExperimentStore();
const showed = ref(new Map<string, boolean>());


function updateColour(logdir: string, e: Event) {
    const colour = (e.target as HTMLInputElement).value;
    const exp = store.experiments.filter(e => e.logdir == logdir)[0];
    if (exp == null || colour == null) return;
    exp.test_metrics.datasets.forEach(ds => ds.colour = colour);
    exp.train_metrics.datasets.forEach(ds => ds.colour = colour);
}

function toggleExperiment(exp: Experiment) {
    if (showed.value.get(exp.logdir)) {
        emits("hide-experiment", exp);
    } else {
        emits("show-experiment", exp);
    }
    showed.value.set(exp.logdir, !showed.value.get(exp.logdir));
}


const emits = defineEmits<{
    (event: "show-experiment", experiment: Experiment): void
    (event: "hide-experiment", experiment: Experiment): void
    (event: "inspect-experiment", experiment: Experiment): void
}>()


const { experiments } = storeToRefs(store);
watch(experiments, (newExperiments, oldExperiments) => {
    // If an experiment has been removed, then hide it
    if (newExperiments.length < oldExperiments.length) {
        const experiment = oldExperiments.filter(exp => !newExperiments.map(e => e.logdir).includes(exp.logdir))[0];
        showed.value.delete(experiment.logdir);
        emits("hide-experiment", experiment)
    }
    // If an experiment has been added, then show it
    else if (newExperiments.length > oldExperiments.length) {
        const experiment = newExperiments.filter(exp => !oldExperiments.map(e => e.logdir).includes(exp.logdir))[0];
        showed.value.set(experiment.logdir, true);
        emits("show-experiment", experiment)
        return;
    }
});
</script>

