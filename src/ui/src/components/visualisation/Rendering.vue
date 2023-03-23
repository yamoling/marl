<template>
    <div class="text-center">
        <div class="row">
            <div class="col-6">
                <h4> Previous </h4>
                <img :src="'data:image/jpg;base64, ' + previousImage" />
            </div>
            <div class="col-6">
                <h4>Current</h4>
                <img :src="'data:image/jpg;base64, ' + currentImage" />
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">

interface Props {
    previousImage: string,
    currentImage: string,
}

defineProps<Props>()
</script>