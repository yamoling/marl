<script setup lang="ts">
/** Runs shelf: Aggregate · Aggregate + runs · Runs only, plus the seed picker. */
import type { PlotSpec, RunsMode } from "../../domain/plot";
import Segmented from "../shell/Segmented.vue";
import SeedPicker from "./SeedPicker.vue";
import type { SeedExperiment } from "./types";

const props = defineProps<{ runs: PlotSpec["runs"]; experiments: SeedExperiment[] }>();
const emit = defineEmits<{ "update:runs": [v: PlotSpec["runs"]] }>();
const MODES = [
    { value: "aggregate" as RunsMode, label: "Aggregate" },
    { value: "aggregate+runs" as RunsMode, label: "Aggregate + runs" },
    { value: "runs" as RunsMode, label: "Runs only" },
];
</script>

<template>
    <div class="shelf" data-shelf="runs">
        <label>Runs</label>
        <Segmented
            :model-value="runs.mode"
            :options="MODES"
            label="Runs"
            @update:model-value="(mode) => emit('update:runs', { ...props.runs, mode })"
        />
        <SeedPicker
            :seeds="runs.seeds"
            :experiments="experiments"
            @update:seeds="(seeds) => emit('update:runs', { ...props.runs, seeds })"
        />
    </div>
</template>
