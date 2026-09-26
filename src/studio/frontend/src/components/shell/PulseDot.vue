<script setup lang="ts">
/** Pulsing "running" dot (Atlas). Static when the user prefers reduced motion. */
withDefaults(defineProps<{ color?: string; size?: number; label?: string }>(), { color: "var(--acc)", size: 7, label: "running" });
</script>

<template>
  <span class="pulse" :style="{ color, width: size + 'px', height: size + 'px' }" role="img" :aria-label="label" />
</template>
