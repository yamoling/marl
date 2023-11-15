<template>
    <div>
        <h4>
            All experiments
            <!-- Reload button -->
            <button class="btn btn-outline-info" @click="() => store.refresh()">
                <font-awesome-icon :icon="['fas', 'sync-alt']" :spin="store.anyLoading" />
            </button>
        </h4>
        <div class="input-group">
            <span class="input-group-text">
                <font-awesome-icon :icon="['fas', 'search']" />
            </span>
            <input class="form-control" type="text" v-model="searchString" />
        </div>
        <table>
            <template v-for="(e, i) in store.experiments">
                <tr v-show="searchMatch(searchString, e.logdir)" class="experiment-row">
                    <td>
                        <font-awesome-icon v-if="!e.runs.every((r => r.pid == null))" :icon="['fas', 'spinner']" spin />
                        {{ e.logdir }}
                    </td>
                    <td>
                        <button class="btn btn-sm btn-success" @click="() => loadExperiment(e.logdir)">
                            <font-awesome-icon v-if="store.loading[i]" :icon="['fas', 'spinner']" spin />
                            <font-awesome-icon v-else :icon="['fas', 'arrow-right']" />
                        </button>
                    </td>
                </tr>
            </template>
        </table>
    </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { searchMatch } from '../../utils';
import { useExperimentStore } from '../../stores/ExperimentStore';

const searchString = ref("");
const store = useExperimentStore();


async function loadExperiment(logdir: string) {
    emits("load-experiment", logdir);
}

const emits = defineEmits<{
    (event: "load-experiment", logdir: string): void
}>()
</script>

