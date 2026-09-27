<script setup lang="ts">
/** Mini curve of a schedule parameter (from `ParamRow.curve`), with start → end values. */
import { computed } from "vue";
import type { ParamRow } from "../../api";
import Sparkline from "../../charts/Sparkline.vue";
import { fmtStep } from "../../domain/format";
import { formatParamValue } from "../../domain/params";

const props = withDefaults(defineProps<{ row: ParamRow; width?: number; height?: number; labels?: boolean }>(), {
    width: 90,
    height: 22,
    labels: false,
});
const curve = computed(() => props.row.curve);
const ends = computed(() => (curve.value ? [curve.value.y[0], curve.value.y[curve.value.y.length - 1]] : null));
const steps = computed(() => (curve.value ? fmtStep(curve.value.x[curve.value.x.length - 1] / 1.25) : ""));
</script>

<template>
    <span class="sched" :title="`${row.cls}`">
        <Sparkline v-if="curve" :x="curve.x" :y="curve.y" color="#0e9f8e" :width="width" :height="height" :fill="true" />
        <span v-if="labels && ends" class="ends"
            >{{ formatParamValue(ends[0]) }} → {{ formatParamValue(ends[1])
            }}<template v-if="row.cls !== 'ConstantSchedule'"> over {{ steps }}</template></span
        >
    </span>
</template>

<style scoped>
.sched {
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.ends {
    font-size: 11px;
    color: var(--ink3);
    white-space: nowrap;
}
</style>
