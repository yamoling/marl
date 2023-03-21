<template>
    <div class="row">
        <div class="col table-scrollable">
            <table class="table table-striped table-hover" style="cursor: pointer;">
                <thead>
                    <tr>
                        <th>
                            <button class="btn btn-sm btn-outline-success" @click="() => store.refresh()">
                                <!-- Reload icon -->
                                <font-awesome-icon icon="fa-solid fa-sync" :spin="store.loading" />
                            </button>
                            Log directory
                        </th>
                        <th> Algorithm </th>
                        <th> Environment </th>
                        <th> Start date </th>
                        <th></th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-for="[logdir, info] in store.experimentInfos" @click="() => loadExperiment(logdir)">
                        <td> {{ logdir }} </td>
                        <td> {{ info.algorithm.name }} </td>
                        <td> {{ info.env.name }} </td>
                        <td> {{ (info.timestamp) ? new Date(info.timestamp) : '' }} </td>
                        <td>
                            <button class="btn btn-sm btn-danger" @click="() => deleteExperiment(logdir, info)">
                                <font-awesome-icon :icon="['fas', 'trash']" />
                            </button>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</template>
<script setup lang="ts">
import { ref } from 'vue';
import { ExperimentInfo } from '../models/Infos';
import { useExperimentStore } from '../stores/ExperimentStore';


const spin = ref(false);
const store = useExperimentStore();

function deleteExperiment(logdir: string, info: ExperimentInfo) {
    console.log(info)
    if (confirm(`Are you sure you want to delete experiment ${logdir}?`)) {
        store.deleteExperiment(logdir);
    }
}

function loadExperiment(logdir: string) {
    store.loadExperiment(logdir);
}
</script>