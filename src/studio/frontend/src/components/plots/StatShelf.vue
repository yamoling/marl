<script setup lang="ts">
/** Statistic shelf: centre (mean/median) and band (ci95/std/min–max/none). No smoothing. */
import type { PlotSpec } from "../../domain/plot";
import Segmented from "../shell/Segmented.vue";

const props = defineProps<{ stat: PlotSpec["stat"]; dim?: boolean }>();
const emit = defineEmits<{ "update:stat": [v: PlotSpec["stat"]] }>();
const CENTERS = [
  { value: "mean" as const, label: "mean" },
  { value: "median" as const, label: "median" },
];
const BANDS = [
  { value: "ci95" as const, label: "ci95", title: "95% confidence interval of the centre" },
  { value: "std" as const, label: "std", title: "± one standard deviation" },
  { value: "minmax" as const, label: "min–max", title: "Range over runs" },
  { value: "none" as const, label: "none" },
];
</script>

<template>
  <div class="shelf" :class="{ dim }" data-shelf="stat" :title="dim ? 'Not used in “Runs only” mode' : undefined">
    <label>Stat</label>
    <Segmented :model-value="stat.center" :options="CENTERS" label="Centre" @update:model-value="(center) => emit('update:stat', { ...props.stat, center })" />
    <Segmented :model-value="stat.band" :options="BANDS" label="Band" @update:model-value="(band) => emit('update:stat', { ...props.stat, band })" />
  </div>
</template>
