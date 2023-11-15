<template>
    <div v-show="store.experimentResults.length > 0" class="mb-3">
        <h3 class="text-center">
            Legend
            <button class="btn btn-outline-info col-auto" :disabled="store.anyLoading">
                <font-awesome-icon :icon="['fas', 'sync']" :spin="store.anyLoading" />
            </button>
        </h3>
        <table>
            <tbody>
                <template v-for="expResult in  store.experimentResults ">
                    <tr class="experiment-row">
                        <td>
                            <input type="color" v-model="expResult.datasets[0].colour"
                                @change="(e) => updateColour(expResult.logdir, e)">
                        </td>
                        <td> {{ expResult.logdir }}</td>
                        <td>
                            <button class="btn btn-sm btn-outline-primary"
                                @click="() => emits('inspect-experiment', expResult.logdir)">
                                <font-awesome-icon :icon="['fas', 'magnifying-glass']" />
                            </button>
                        </td>
                        <td>
                            <button class="btn btn-sm btn-outline-info" @click="() => toggle(expResult.logdir)">
                                <font-awesome-icon v-if="showed.get(expResult.logdir)" :icon="['fas', 'eye']" />
                                <font-awesome-icon v-else :icon="['fas', 'eye-slash']" />
                            </button>
                        </td>
                        <td>
                            <button class="btn btn-sm btn-outline-danger"
                                @click="() => store.unloadExperimentResults(expResult.logdir)">
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
import { ref } from 'vue';
import { useDatasetStore } from '../../stores/DatasetStore';

const store = useDatasetStore();
const showed = ref(new Map<string, boolean>());

function updateColour(logdir: string, e: Event) {
    const target = e.target as HTMLInputElement;
    store.setDatasetColour(logdir, target.value);
}

function toggle(logdir: string) {
    if (showed.value.get(logdir)) {
        emits("hide-experiment", logdir);
    } else {
        emits("show-experiment", logdir);
    }
    showed.value.set(logdir, !showed.value.get(logdir));
}



const emits = defineEmits<{
    (event: "show-experiment", logdir: string): void
    (event: "hide-experiment", logdir: string): void
    (event: "inspect-experiment", logdir: string): void
}>()


// const { experiments } = storeToRefs(store);
// watch(experiments, (newExperiments, oldExperiments) => {
//     // If an experiment has been removed, then hide it
//     if (newExperiments.length < oldExperiments.length) {
//         const experiment = oldExperiments.filter(exp => !newExperiments.map(e => e.logdir).includes(exp.logdir))[0];
//         showed.value.delete(experiment.logdir);
//         emits("hide-experiment", experiment)
//     }
//     // If an experiment has been added, then show it
//     else if (newExperiments.length > oldExperiments.length) {
//         const experiment = newExperiments.filter(exp => !oldExperiments.map(e => e.logdir).includes(exp.logdir))[0];
//         showed.value.set(experiment.logdir, true);
//         emits("show-experiment", experiment)
//         return;
//     }
// });
</script>

../../stores/ResultsStore