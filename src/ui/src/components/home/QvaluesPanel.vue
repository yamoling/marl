<template>
    <section class="selector-panel">
        <div class="selector-toolbar">
            <div class="input-group">
            <span class="input-group-text">
                <font-awesome-icon :icon="['fas', 'search']" />
            </span>
                <input class="form-control" type="text" v-model="searchString" placeholder="Search q-values" />
                <button class="btn btn-outline-secondary input-group-btn" @click="searchString = ''"
                    :disabled="searchString.length === 0">
                <font-awesome-icon :icon="['fas', 'times']" />
            </button>
        </div>

            <div class="selector-actions">
                <button class="btn btn-sm btn-outline-primary" @click="selectFiltered"
                    :disabled="filteredQvalues.length === 0">
                    Select filtered
                </button>
                <button class="btn btn-sm btn-outline-secondary" @click="clearFiltered"
                    :disabled="selectedFilteredCount === 0">
                    Clear filtered
                </button>
                <button class="btn btn-sm btn-outline-danger" @click="clearSelectedQvalues"
                    :disabled="selectedQvalues.length === 0">
                    Reset all
                </button>
            </div>
        </div>

        <div class="selection-summary">
            <span>{{ selectedQvalues.length }} selected</span>
            <span>{{ filteredQvalues.length }} visible</span>
        </div>

        <div class="selector-columns">
            <div v-for="qvalues in qvaluesByColumn" :key="qvalues.join('-')" class="selector-column">
                <ul>
                    <li v-for="qvalueName in qvalues" :key="qvalueName">
                        <label>
                            <input type="checkbox" class="form-check-input"
                                :checked="selectedQvalues.includes(qvalueName)"
                                @change="() => toggleQvalue(qvalueName)">
                            {{ qvalueName }}
                        </label>
                    </li>
                </ul>
            </div>
        </div>
    </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { useQvaluesStore } from '../../stores/QvaluesStore';
import { searchMatch } from '../../utils';

const N_COLS = 4;
const props = defineProps<{
    qvalues: Set<string>
}>();
const searchString = ref("");

const qvaluesStore = useQvaluesStore();
const selectedQvalues = computed(() => qvaluesStore.getSelectedQvalues());
const filteredQvalues = computed(() => Array.from(props.qvalues).filter(m => searchMatch(searchString.value, m)).sort());
const selectedFilteredCount = computed(() => filteredQvalues.value.filter(label => selectedQvalues.value.includes(label)).length);
const qvaluesByColumn = computed(() => {
    const res = [] as string[][];
    for (let i = 0; i < N_COLS; i++) {
        res.push([]);
    }
    filteredQvalues.value.forEach((m, i) => {
        res[i % N_COLS].push(m);
    });
    return res;
});




const emits = defineEmits<{
    (event: "change-selected-qvalues", value: string[]): void
}>();

function clearSelectedQvalues() {
    qvaluesStore.clearSelectedQvalues();
    emits("change-selected-qvalues", selectedQvalues.value);
}

function selectFiltered() {
    filteredQvalues.value.forEach((label) => {
        if (!selectedQvalues.value.includes(label)) {
            qvaluesStore.addSelectedQvalue(label);
        }
    });
    emits("change-selected-qvalues", selectedQvalues.value);
}

function clearFiltered() {
    filteredQvalues.value.forEach((label) => {
        if (selectedQvalues.value.includes(label)) {
            qvaluesStore.removeSelectedQvalue(label);
        }
    });
    emits("change-selected-qvalues", selectedQvalues.value);
}

function toggleQvalue(qvalueName: string) {
    if (selectedQvalues.value.includes(qvalueName)) {
        qvaluesStore.removeSelectedQvalue(qvalueName);
    } else {
        qvaluesStore.addSelectedQvalue(qvalueName);
    }
    emits("change-selected-qvalues", selectedQvalues.value);
}

onMounted(() => {
    emits("change-selected-qvalues", selectedQvalues.value);
})


</script>
<style scoped>
.selector-panel {
    display: grid;
    gap: 0.65rem;
}

.selector-toolbar {
    display: flex;
    gap: 0.75rem;
    align-items: center;
}

.selector-toolbar .input-group {
    flex: 1;
}

.selector-actions {
    display: flex;
    gap: 0.5rem;
}

.selection-summary {
    display: flex;
    gap: 1rem;
    font-size: 0.84rem;
    color: var(--bs-secondary-color);
}

.selector-columns {
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
}

.selector-column {
    flex: 1;
    min-width: 0;
}

ul {
    margin: 0;
    padding-left: 1.1rem;
}

li {
    margin-bottom: 0.25rem;
}

label {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    cursor: pointer;
}
</style>