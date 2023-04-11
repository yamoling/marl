<template>
    <div class="row">
        <div class="col table-scrollable">
            <table class="table table-striped table-hover">
                <thead>
                    <tr>
                        <th class="text-center"> Status </th>
                        <th class="sortable" @click="() => sortBy('logdir')">
                            Log directory
                            <font-awesome-icon class="px-2" :icon="['fas', 'sort']" />
                        </th>
                        <th class="sortable" @click="() => sortBy('env')">
                            Environment
                            <font-awesome-icon class="px-2" :icon="['fas', 'sort']" />
                        </th>
                        <th class="sortable" @click="() => sortBy('algo')">
                            Algorithm
                            <font-awesome-icon class="px-2" :icon="['fas', 'sort']" />
                        </th>
                        <th> Train/Test policies</th>
                        <th class="sortable" @click="() => sortBy('date')">
                            Start date
                            <font-awesome-icon class="px-2" :icon="['fas', 'sort']" />
                        </th>
                        <th></th>
                    </tr>
                </thead>
                <tbody style="cursor: pointer;">
                    <tr v-for="info in experimentInfos" @click="() => loadExperiment(info.logdir)">
                        <td class="text-center">
                            <font-awesome-icon v-if="info.runs.every((r => r.pid == null))" :icon="['fas', 'check']" />
                            <font-awesome-icon v-else :icon="['fas', 'spinner']" spin />
                        </td>
                        <td> {{ info.logdir }} </td>
                        <td> {{ info.env.name }} </td>
                        <td> {{ info.algorithm.name }} </td>
                        <td> {{ info.algorithm.train_policy.name }} / {{ info.algorithm.test_policy.name }} </td>
                        <td> {{ (info.timestamp_ms) ? new Date(info.timestamp_ms).toLocaleString() : '' }} </td>
                        <td>
                            <button class="btn btn-sm btn-danger" :disabled="deleting.includes(info.logdir)"
                                @click.stop="() => deleteExperiment(info.logdir)" title="Delete experiment">
                                <font-awesome-icon v-if="deleting.includes(info.logdir)" :icon="['fas', 'spinner']" spin />
                                <font-awesome-icon v-else="" :icon="['fas', 'trash']" />
                            </button>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
        <div class="col-12 mt-2">
            <div class="row text-center mx-5 px-5">
                <button class="col mx-2 btn btn-info" @click="() => emits('compare-experiments')">
                    Compare experiments
                    <font-awesome-icon class="ps-2" :icon="['fas', 'chart-line']" />
                </button>
                <button class="col mx-2 btn btn-success" @click="() => emits('create-experiment')">
                    Create a new experiment
                    <font-awesome-icon class="ps-2" :icon="['fas', 'plus']" />
                </button>
                <button class="col mx-2 btn btn-outline-info" @click.stop="() => store.refresh()">
                    Refresh
                    <font-awesome-icon icon="fa-solid fa-sync" :spin="store.loading" />
                </button>
            </div>
        </div>
    </div>
</template>
<script setup lang="ts">
import { computed, ref } from 'vue';
import { useExperimentStore } from '../stores/ExperimentStore';

const store = useExperimentStore();
const deleting = ref([] as string[]);
const sortKey = ref("logdir" as "logdir" | "env" | "algo" | "date");
const sortOrder = ref("ASCENDING" as "ASCENDING" | "DESCENDING");


function deleteExperiment(logdir: string) {
    if (confirm(`Are you sure you want to delete experiment ${logdir}?`)) {
        deleting.value.push(logdir);
        store.deleteExperiment(logdir).then(() => deleting.value = deleting.value.filter(x => x !== logdir));
        emits("experiment-deleted", logdir);
    }
}

function loadExperiment(logdir: string) {
    emits("experiment-selected", logdir);
}

const emits = defineEmits<{
    (event: "experiment-selected", logdir: string): void
    (event: "experiment-deleted", logdir: string): void
    (event: "create-experiment"): void
    (event: "compare-experiments"): void
}>();

const experimentInfos = computed(() => {
    const entries = [...store.experimentInfos];
    switch (sortKey.value) {
        case "logdir":
            entries.sort((a, b) => a.logdir.localeCompare(b.logdir));
            break;
        case "env":
            entries.sort((a, b) => a.env.name.localeCompare(b.env.name));
            break;
        case "algo":
            entries.sort((a, b) => a.algorithm.name.localeCompare(b.algorithm.name));
            break;
        case "date":
            entries.sort((a, b) => a.timestamp_ms - b.timestamp_ms);
            break;
    }
    if (sortOrder.value === "DESCENDING") {
        entries.reverse();
    }
    return entries;
});

function sortBy(key: "logdir" | "env" | "algo" | "date") {
    if (sortKey.value === key) {
        sortOrder.value = sortOrder.value === "ASCENDING" ? "DESCENDING" : "ASCENDING";
    } else {
        sortKey.value = key;
        sortOrder.value = "ASCENDING";
    }
}

</script>
<style scoped>
th.sortable:hover {
    cursor: pointer;
    text-decoration: underline;
}
</style>
