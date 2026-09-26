<script setup lang="ts">
import EpisodesSheet from "./components/replay/EpisodesSheet.vue";
import Toasts from "./components/shell/Toasts.vue";
import TooltipLayer from "./components/shell/TooltipLayer.vue";
</script>

<template>
    <RouterView />
    <EpisodesSheet />
    <Toasts />
    <TooltipLayer />
</template>
