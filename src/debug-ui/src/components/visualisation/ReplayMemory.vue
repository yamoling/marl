<template>
    <div>
        <div class=" progress-stacked mb-2 col-10" style="height: 40px;">
            <div v-for="prio, i in priorities" class="progress" style="height: 100%;"
                :style="{ width: `${(prio * 100) / cumsum}%` }" @click="() => changeSelected(i)">
                <div v-if="selected == i" class="progress-bar" :class="COLORS[i % COLORS.length]"
                    style="border: 1px solid;"> </div>
                <div v-else class="progress-bar" :class="COLORS[i % COLORS.length]"> </div>
            </div>
        </div>
        <div class="row">
            <div class="mx-auto col-auto">
                <div class="input-group">
                    <button type="button" class="btn btn-success" @click="() => changeSelected(selected-1)">
                        <font-awesome-icon icon="fa-solid fa-solid fa-backward-step" />
                    </button>
                    <span class="input-group-text"> Selected {{ selected }} </span>
                    <button type="button" class="btn btn-success" @click="() => changeSelected(selected+1)">
                        <font-awesome-icon icon="fa-solid fa-solid fa-forward-step" />
                    </button>
                </div>
            </div>
        </div>
        <div class="row mb-2">
            <div class="col-auto">
                <img :src="'data:image/jpg;base64, ' + previousImage">
            </div>
            <div class="col-auto">
                <img :src="'data:image/jpg;base64, ' + currentImage">
            </div>
        </div>
        <button role="button" class="btn btn-primary" @click="loadPriorities">
            Get priorities
        </button>
    </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useMemoryStore } from '../../stores/MemoryStore';

const COLORS = [
    'bg-primary',
    'bg-secondary',
    'bg-success',
    'bg-danger',
    'bg-warning',
    'bg-info',
    'bg-light',
    'bg-dark',
] as const;

const memorySize = ref(0);
const priorities = ref([] as number[]);
const selected = ref(0);
const currentImage = ref("");
const previousImage = ref("");
const cumsum = ref(0);
const store = useMemoryStore();


function loadPriorities() {
    store.getPriorities().then(p => {
        priorities.value = p.priorities;
        cumsum.value = p.cumsum;
        memorySize.value = p.priorities.length;
    });
}

function changeSelected(newSelectedIndex: number) {
    selected.value = newSelectedIndex;
    store.getTransition(newSelectedIndex).then(t => {
        console.log(t);
        currentImage.value = t.current_frame;
        previousImage.value = t.prev_frame;
    });
}
</script>