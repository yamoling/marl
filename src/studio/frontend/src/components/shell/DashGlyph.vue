<script setup lang="ts">
/** A short line sample showing a colour and dash pattern (legend, table headers, Y pills). */
withDefaults(defineProps<{ dash?: string; color?: string; width?: number; stroke?: number }>(), { dash: "", color: "currentColor", width: 18, stroke: 2 });
</script>

<template>
  <svg class="dash-glyph" :width="width" height="8" aria-hidden="true">
    <line x1="1" y1="4" :x2="width - 1" y2="4" :stroke="color" :stroke-width="stroke" :stroke-dasharray="dash" stroke-linecap="round" />
  </svg>
</template>

<style scoped>
.dash-glyph {
  flex: none;
}
</style>
