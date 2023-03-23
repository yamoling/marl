<template>
    <div class="row">
        <div class="col table-scrollable">
            <table class="table table-striped table-hover">
                <thead>
                    <tr>
                        <th>
                            <button class="btn btn-sm btn-outline-success me-2" @click="() => store.refresh()">
                                <!-- Reload icon -->
                                <font-awesome-icon icon="fa-solid fa-sync" :spin="store.loading" />
                            </button>
                            Log directory
                        </th>
                        <th> Environment </th>
                        <th> Algorithm </th>
                        <th> Train/Test policies</th>
                        <th> Start date </th>
                        <th></th>
                    </tr>
                </thead>
                <tbody style="cursor: pointer;">
                    <tr v-for="[logdir, info] in store.experimentInfos" @click="() => selectExperiment(logdir)">
                        <td> {{ logdir }} </td>
                        <td> {{ info.env.name }} </td>
                        <td> {{ info.algorithm.name }} </td>
                        <td> {{ info.algorithm.train_policy.name }} / {{ info.algorithm.test_policy.name }} </td>
                        <td> {{ (info.timestamp_ms) ? new Date(info.timestamp_ms * 1000).toLocaleString() : '' }} </td>
                        <td>
                            <button class="btn btn-sm btn-danger" :disabled="deleting.includes(logdir)"
                                @click.stop="() => deleteExperiment(logdir)" title="Delete experiment">
                                <font-awesome-icon v-if="deleting.includes(logdir)" :icon="['fas', 'spinner']" spin />
                                <font-awesome-icon v-else="" :icon="['fas', 'trash']" />
                            </button>
                        </td>
                    </tr>
                    <tr>
                        <td colspan="6">
                            <!-- Add button -->
                            <button class="btn btn-sm btn-success px-4" @click="() => emits('create-experiment')">
                                <font-awesome-icon :icon="['fas', 'plus']" />
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
import { useExperimentStore } from '../stores/ExperimentStore';

const store = useExperimentStore();
const deleting = ref([] as string[]);

function deleteExperiment(logdir: string) {
    if (confirm(`Are you sure you want to delete experiment ${logdir}?`)) {
        deleting.value.push(logdir);
        store.deleteExperiment(logdir).then(() => deleting.value = deleting.value.filter(x => x !== logdir));
        emits("experiment-deleted", logdir);
    }
}

function selectExperiment(logdir: string) {
    emits("experiment-selected", logdir);
}

const emits = defineEmits(["experiment-selected", "experiment-deleted", "create-experiment"]);
</script>