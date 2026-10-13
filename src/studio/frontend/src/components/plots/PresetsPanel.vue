<script setup lang="ts">
/** Applicable starting plots for an empty workspace. */
import { computed } from "vue";
import { presets } from "../../domain/plot";
import { useExperimentsStore } from "../../stores/experiments";
import { useWorkspaceStore } from "../../stores/workspace";

const experiments = useExperimentsStore();
const workspace = useWorkspaceStore();
const available = computed(() => presets(experiments.context));
</script>

<template>
  <section v-if="available.length" class="presets" aria-label="Plot presets">
    <span class="muted">Start with a preset</span>
    <button v-for="preset in available" :key="preset.title" type="button" class="btn" @click="workspace.createPlot(preset)">
      {{ preset.title }}
    </button>
  </section>
</template>

<style scoped>
.presets { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-bottom: 18px; }
</style>
