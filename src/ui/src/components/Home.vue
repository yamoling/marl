<template>
    <div class="row">
        <div class="col table-scrollable">
            <table class="table table-striped table-hover">
                <thead>
                    <tr>
                        <th> Status </th>
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
                    <tr v-for="[logdir, info] in experimentInfos" @click="() => selectExperiment(logdir)">
                        <th>
                            <font-awesome-icon v-if="info.runs.every((r => r.pid == null))" :icon="['fas', 'check']" />
                            <font-awesome-icon v-else :icon="['fas', 'spinner']" spin />
                        </th>
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
                    <tr class="text-center">
                        <td colspan="7">
                            <!-- Add button -->
                            <button style="width: 25%;" class="btn btn-success me-2"
                                @click="() => emits('create-experiment')">
                                Create a new experiment
                                <font-awesome-icon class="ps-2" :icon="['fas', 'plus']" />
                            </button>
                            <button style="width: 25%" class="btn btn-outline-success ms-2"
                                @click.stop="() => store.refresh()">
                                Refresh
                                <font-awesome-icon icon="fa-solid fa-sync" :spin="store.loading" />
                            </button>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</template>
<script setup lang="ts">
import { computed, ref } from 'vue';
import { ExperimentInfo } from '../models/Infos';
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

function selectExperiment(logdir: string) {
    emits("experiment-selected", logdir);
}

const emits = defineEmits(["experiment-selected", "experiment-deleted", "create-experiment"]);

const experimentInfos = computed(() => {
    const entries = [] as [string, ExperimentInfo][];
    store.experimentInfos.forEach((value, key) => entries.push([key, value]));
    switch (sortKey.value) {
        case "logdir":
            entries.sort((a, b) => a[0].localeCompare(b[0]));
            break;
        case "env":
            entries.sort((a, b) => a[1].env.name.localeCompare(b[1].env.name));
            break;
        case "algo":
            entries.sort((a, b) => a[1].algorithm.name.localeCompare(b[1].algorithm.name));
            break;
        case "date":
            entries.sort((a, b) => a[1].timestamp_ms - b[1].timestamp_ms);
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
