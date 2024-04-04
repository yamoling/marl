<template>
    <TestOnOtherEnvironment ref="modal" />
    <div ref="contextMenu" class="context-menu">
        <ul>
            <li @click="() => rename()">
                <font-awesome-icon :icon="['far', 'pen-to-square']" class="pe-2" />
                Rename
            </li>
            <li @click="() => remove()">
                <font-awesome-icon :icon="['fa', 'trash']" class="text-danger pe-2" />
                Delete
            </li>
            <li @click="() => modal.showModal(clickedExperiment)">
                <font-awesome-icon :icon="['fa', 'play']" class=" text-success pe-2" />
                Test on other env
            </li>
            <li>
                <font-awesome-icon :icon="['fas', 'person-running']" class="pe-2" />
                Start a new run
            </li>
        </ul>
    </div>
</template>
<script setup lang="ts">
import { ref } from 'vue';
import { useExperimentStore } from '../../stores/ExperimentStore';
import TestOnOtherEnvironment from '../modals/TestOnOtherEnv.vue';

const contextMenu = ref({} as HTMLDivElement);
const modal = ref({} as typeof TestOnOtherEnvironment)
const clickedExperiment = ref({} as Experiment);
const experimentStore = useExperimentStore();

document.addEventListener('click', () => {
    contextMenu.value.style.display = 'none';
});

function show(exp: Experiment, x: number, y: number) {
    clickedExperiment.value = exp;
    contextMenu.value.style.left = `${x}px`;
    contextMenu.value.style.top = `${y}px`;
    contextMenu.value.style.display = 'block';
}

defineExpose({ show });


function rename() {
    const logdir = clickedExperiment.value.logdir;
    const newLogdir = prompt("Enter new name for the experiment", logdir);
    if (newLogdir === null) return;
    experimentStore.rename(logdir, newLogdir);
}

function remove() {
    const logdir = clickedExperiment.value.logdir;
    if (confirm(`Are you sure you want to delete the experiment ${logdir}?`)) {
        experimentStore.remove(logdir);
    }
}
</script>