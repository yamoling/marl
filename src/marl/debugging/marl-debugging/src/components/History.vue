<template>
    <Rendering :current-image="currentImage" :previous-image="previousImage" :reward="reward"></Rendering>
    <div class="container">
        <button>Prev</button>
        <button>Next</button>
    </div>
</template>


<script setup lang="ts">
import { ref } from 'vue';
import { ServerUpdate } from '../models/ServerUpdate';

const history = [] as ServerUpdate[];
let currentStep = 0;
const currentImage = ref("");
const previousImage = ref("");


function push(update: ServerUpdate) {
    history.push(update);
    currentStep = history.length - 1;
    if (history.length > 100) {
        history.shift();
    }
}


function next() {
    currentStep++;
    currentImage.value = history[currentStep].b64_rendering;
    previousImage.value = history[currentStep - 1].b64_rendering;
}

function prev() {
    currentStep--;
    currentImage.value = history[currentStep].b64_rendering;
    previousImage.value = history[currentStep - 1].b64_rendering;
}

defineExpose({ prev, next, push });

</script>