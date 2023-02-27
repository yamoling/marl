<template>
    <div>
        <h4> Replay memory</h4>
        Cumulative sum: {{ cumsum.toFixed(2) }} <br />
        <span v-if="selected >= 0"> Selected value: {{ priorities[selected].toFixed(3) }}</span>
        <label> Number of lines
            <input type="text" v-model="numberOfLines" size="5">
        </label>

        <div class="col-12 row mb-1" style="border: 1px solid">
            <div v-for="prio, i in priorities" class="p-0 text-center" style="height: 50px; overflow: hidden;" :style="{
                width: `${(prio * 100 * numberOfLines) / cumsum}%`, backgroundColor: '#' + rainbow.colourAt(prio),
                border: (selected == i) ? '1px solid' : 'none'
            }" @click="() => changeSelected(i)">
                {{ prio.toFixed(3) }}
            </div>
        </div>

        <div class="row mx-auto mb-2">
            <div class="col-auto">
                <div class="input-group">
                    <button type="button" class="btn btn-success" @click="() => changeSelected(selected - 1)">
                        <font-awesome-icon icon="fa-solid fa-solid fa-backward-step" />
                    </button>
                    <span class="input-group-text"> Selected {{ selected }} </span>
                    <button type="button" class="btn btn-success" @click="() => changeSelected(selected + 1)">
                        <font-awesome-icon icon="fa-solid fa-solid fa-forward-step" />
                    </button>
                </div>
            </div>
            <div class="col-auto">
                <button class="btn btn-primary" @click="loadPriorities">
                    <!-- Reload icon -->
                    <font-awesome-icon icon="fa-solid fa-solid fa-sync" :class="loadingPriorities ? 'spin' : ''" />
                </button>
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
    </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useMemoryStore } from '../../stores/MemoryStore';
import Rainbow from "rainbowvis.js";

const rainbow = new Rainbow();
rainbow.setSpectrum("deepskyblue", "turquoise", "lightgreen", "limegreen");

const loadingPriorities = ref(false);
const numberOfLines = ref(10);
const memorySize = ref(0);
const priorities = ref([] as number[]);
const selected = ref(-1 as number);
const currentImage = ref("");
const previousImage = ref("");
const cumsum = ref(0);
const store = useMemoryStore();

function loadPriorities() {
    selected.value = -1;
    loadingPriorities.value = true;
    store.getPriorities().then(p => {
        rainbow.setNumberRange(0, Math.max(...p.priorities));
        priorities.value = p.priorities;
        cumsum.value = p.cumsum;
        memorySize.value = p.priorities.length;
        loadingPriorities.value = false;
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