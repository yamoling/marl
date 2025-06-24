<template>
    <div class="row">
        <h3 class="text-center">Qvalues</h3>
        <div class="input-group pb-2">
            <span class="input-group-text">
                <font-awesome-icon :icon="['fas', 'search']" />
            </span>
            <input class="form-control" type="text" v-model="searchString" />
            <!-- Cross icon to delete the search string -->
            <button class="btn btn-secondary input-group-btn" @click="searchString = ''">
                <font-awesome-icon :icon="['fas', 'times']" />
            </button>
        </div>
        <div class="container">
            <div v-for="qvalues in qvaluesByColumn">
                <ul>
                    <li v-for="qvalueName in qvalues">
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
        <button class="btn btn-outline-danger" @click="clearSelectedQvalues">
            Reset selection
            <font-awesome-icon :icon="['fas', 'trash']" />
        </button>
    </div>
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
const qvaluesByColumn = computed(() => {
    // N columns
    const res = [] as string[][];
    for (let i = 0; i < N_COLS; i++) {
        res.push([]);
    }
    Array.from(props.qvalues).filter(m => searchMatch(searchString.value, m)).sort().forEach((m, i) => {
        res[i % N_COLS].push(m);
    });
    // Array.from(props.qvalues).sort().forEach((m, i) => {
    //     res[i % N_COLS].push(m);
    // });
    return res;
});




const emits = defineEmits<{
    (event: "change-selected-qvalues", value: string[]): void
}>();

function clearSelectedQvalues() {
    qvaluesStore.clearSelectedQvalues();
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
.container {
    display: flex;
    justify-content: space-around;
}
</style>