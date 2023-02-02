<template>
    <div class="text-center">
        <h2>Rendering</h2>
        Reward: {{ reward?.toFixed(4) }}
        <div class="row">
            <div class="col-6">
                <h4>
                    <input type="checkbox" name="showPrevious" v-model="showPrevious">
                    Previous
                </h4>
                <img v-if="showPrevious" :src="'data:image/jpg;base64, ' + previousImage" />
            </div>
            <div class="col-6">
                <h4>Current</h4>
                <img :src="'data:image/jpg;base64, ' + currentImage" />
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">import { ref } from 'vue';

interface Props {
    previousImage: string,
    currentImage: string,
    reward: number | null
}

const showPrevious = ref(true);

defineProps<Props>()
</script>