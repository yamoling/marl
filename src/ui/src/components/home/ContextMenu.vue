<template>
    <TestOnOtherEnvironment ref="modal" />
    <NewRun ref="newRunModal" />
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
            <li @click="archive">
                <font-awesome-icon :icon="['fas', 'box-archive']" class="pe-2" />
                Archive
            </li>
            <li @click="() => newRunModal.showModal(clickedExperiment)">
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
import NewRun from '../modals/NewRun.vue';
import { Experiment } from '../../models/Experiment';

const contextMenu = ref({} as HTMLDivElement);
const modal = ref({} as typeof TestOnOtherEnvironment)
const newRunModal = ref({} as typeof NewRun)
const clickedExperiment = ref({} as Experiment);
const experimentStore = useExperimentStore();

document.addEventListener('click', () => {
    if (contextMenu.value) {
        contextMenu.value.style.display = 'none';
    }
});

// Escape key also closes the context menu
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        contextMenu.value.style.display = 'none';
    }
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

function archive() {
    const currentLogdir = clickedExperiment.value.logdir;
    const newLogdir = currentLogdir.replace("logs/", "archives/")
    console.log(newLogdir);
    experimentStore.rename(currentLogdir, newLogdir);
}

</script>


<style>
.context-menu {
    width: fit-content;
    position: fixed;
    display: none;
    background-color: #fff;
    border: 1px solid #ccc;
    padding: 5px;
    z-index: 1000;
}

.context-menu ul {
    list-style: none;
    padding: 0;
    margin: 0;
}

.context-menu ul li {
    padding: 5px 10px;
    cursor: pointer;
}

.context-menu ul li:hover {
    background-color: #f0f0f0;
}
</style>