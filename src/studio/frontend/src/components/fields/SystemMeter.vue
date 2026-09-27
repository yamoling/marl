<script setup lang="ts">
/**
 * System meter (Atlas) pinned at the bottom of the fields panel: CPU, RAM and one bar per GPU,
 * amber above 75 %. Clicking opens GPU memory details.
 */
import { ref } from "vue";
import { vTip } from "../../composables/tooltip";
import { HOT_THRESHOLD, useSystemStore } from "../../stores/system";
import Popover from "../shell/Popover.vue";

const system = useSystemStore();
const open = ref(false);
const root = ref<HTMLButtonElement | null>(null);
const pct = (v: number) => `${Math.round(Math.max(0, Math.min(100, v)))}%`;
</script>

<template>
  <button ref="root" class="sysm" type="button" aria-label="System usage details" @click="open = !open">
    <div v-if="!system.hasReading" class="muted wait">{{ system.state === "paused" ? "System readings unavailable" : "Reading system usage…" }}</div>
    <div v-for="r in system.rows" :key="r.key" class="sys-row" v-tip="r.tip">
      <span>{{ r.key }}</span>
      <div class="sys-bar"><i :class="{ hot: r.value > HOT_THRESHOLD }" :style="{ width: pct(r.value) }" /></div>
      <span class="v">{{ pct(r.value) }}</span>
    </div>
  </button>
  <Popover v-model:open="open" :anchor="root" placement="top-start" :min-width="280" label="System usage">
    <h4>System</h4>
    <div class="line"><span>CPU</span><b>{{ pct(system.cpu) }}</b></div>
    <div class="line"><span>RAM</span><b>{{ pct(system.ram) }}</b></div>
    <template v-for="g in system.gpus" :key="g.index">
      <hr />
      <h4>GPU {{ g.index }}</h4>
      <div class="line"><span>Utilisation</span><b>{{ pct(g.utilization) }}</b></div>
      <div class="line"><span>Memory</span><b>{{ Math.round(g.usedMB).toLocaleString("en-US") }} / {{ Math.round(g.totalMB).toLocaleString("en-US") }} MB</b></div>
      <div class="sys-bar big"><i :class="{ hot: g.memory > HOT_THRESHOLD }" :style="{ width: pct(g.memory) }" /></div>
      <div class="line muted small">Runs on this GPU: not reported by the server yet</div>
    </template>
    <div v-if="!system.gpus.length" class="line muted">No GPU detected</div>
  </Popover>
</template>

<style scoped>
.sysm {
  display: flex;
  flex-direction: column;
  gap: 5px;
  width: 100%;
  border: 1px solid var(--line);
  background: var(--card);
  border-radius: 12px;
  padding: 8px 10px;
  text-align: left;
}
.sysm:hover {
  border-color: var(--line-hover);
}
.wait {
  font-size: 11px;
}
.v {
  text-align: right;
  color: var(--ink3);
  font-weight: 500;
  font-variant-numeric: tabular-nums;
}
.line {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 2px 10px;
  font-size: 12px;
}
.small {
  font-size: 11px;
}
.sys-bar.big {
  margin: 4px 10px;
  height: 6px;
}
</style>
