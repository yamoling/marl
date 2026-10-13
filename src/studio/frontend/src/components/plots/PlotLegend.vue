<script setup lang="ts">
/**
 * Legend: one toggle per series (click hides/shows, persisted), greyed entries with "Load" for
 * referenced but unloaded experiments, and the colour-by-parameter legend (gradient for numeric
 * values, swatches for categorical ones, ∅ for experiments without the parameter).
 */
import { computed } from "vue";
import { vTip } from "../../composables/tooltip";
import { MISSING_COLOUR, rampColour } from "../../domain/colour";
import { formatParamValue } from "../../domain/params";
import type { LegendEntry, ParamLegend } from "../../domain/plot";
import DashGlyph from "../shell/DashGlyph.vue";

const props = defineProps<{ legend: LegendEntry[]; paramLegend: ParamLegend | null }>();
const emit = defineEmits<{ toggle: [key: string]; load: [id: string] }>();
const gradient = `linear-gradient(90deg, ${rampColour(0)}, ${rampColour(0.5)}, ${rampColour(1)})`;
const numericRange = computed(() => {
  const e = props.paramLegend?.entries ?? [];
  return e.length ? [e[0].value, e[e.length - 1].value] : null;
});
const fmt = (v: unknown) => (typeof v === "number" ? formatParamValue(v) : v === null ? "null" : String(v));
</script>

<template>
  <div class="legend">
    <template v-for="l in legend" :key="l.key">
      <span v-if="l.unloaded" class="lg unloaded">
        <DashGlyph :color="MISSING_COLOUR" :width="22" :stroke="1.6" />{{ l.label }}
        <button type="button" class="btn small" @click="emit('load', l.experiment)">Load</button>
      </span>
      <button v-else type="button" class="lg" :class="{ off: l.hidden }" :aria-pressed="!l.hidden" v-tip="`Click to ${l.hidden ? 'show' : 'hide'}`" @click="emit('toggle', l.key)">
        <DashGlyph :dash="l.dash" :color="l.color" :width="22" :stroke="l.thick ? 2.5 : 1.6" />{{ l.label }}<template v-if="l.axis === 'right'"> (R)</template>
      </button>
    </template>
    <span v-if="paramLegend" class="plegend" :title="paramLegend.path">
      <b>{{ paramLegend.label }}</b>
      <template v-if="paramLegend.numeric && numericRange && paramLegend.entries.length > 2">
        <span>{{ fmt(numericRange[0]) }}</span><i class="grad" :style="{ background: gradient }" /><span>{{ fmt(numericRange[1]) }}</span>
        <span class="muted">({{ paramLegend.entries.length }} values)</span>
      </template>
      <template v-else>
        <span v-for="e in paramLegend.entries" :key="String(e.value)"><i :style="{ background: e.colour }" />{{ fmt(e.value) }}</span>
      </template>
      <span v-if="paramLegend.missing"><i :style="{ background: MISSING_COLOUR }" />∅ absent</span>
    </span>
  </div>
</template>

<style scoped>
.lg.unloaded {
  opacity: 0.7;
  cursor: default;
}
.lg.unloaded .btn {
  margin-left: 4px;
  padding: 0 7px;
}
.plegend {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-left: 4px;
  padding: 2px 8px;
  border-radius: 7px;
  background: var(--param-soft);
  color: var(--param-ink);
  font-size: 11px;
  flex-wrap: wrap;
}
.plegend i {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 3px;
  margin-right: 3px;
  vertical-align: -1px;
}
.plegend i.grad {
  width: 70px;
  height: 8px;
  margin: 0;
}
</style>
