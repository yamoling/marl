<script setup lang="ts">
/** Small inline chart (library cards, minimized plots): line, optional CI band or area fill. */
import { computed } from "vue";
import { sparklinePaths } from "./render";

const props = withDefaults(
  defineProps<{
    x: number[];
    y: (number | null)[];
    lo?: (number | null)[] | null;
    hi?: (number | null)[] | null;
    color?: string;
    width?: number;
    height?: number;
    fill?: boolean;
  }>(),
  { lo: null, hi: null, color: "#4e79a7", width: 120, height: 28, fill: true },
);

const paths = computed(() => sparklinePaths(props.x, props.y, { w: props.width, h: props.height, lo: props.lo, hi: props.hi, fill: props.fill }));
</script>

<template>
  <svg class="sparkline" :width="width" :height="height" :viewBox="`0 0 ${width} ${height}`" aria-hidden="true">
    <path v-if="paths.band" :d="paths.band" :fill="color" fill-opacity="0.15" />
    <path v-if="paths.area" :d="paths.area" :fill="color" fill-opacity="0.12" />
    <path v-if="paths.line" :d="paths.line" fill="none" :stroke="color" stroke-width="1.5" />
  </svg>
</template>

<style scoped>
.sparkline {
  display: block;
  flex: none;
}
</style>
