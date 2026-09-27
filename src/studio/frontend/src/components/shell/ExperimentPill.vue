<script setup lang="ts">
/**
 * Top-bar pill of a loaded experiment: colour dot (pulsing when running), short name, health
 * mark with tooltip, running percentage with a shimmering progress bar along the bottom edge,
 * and × to unload.
 */
import { computed } from "vue";
import { vTip } from "../../composables/tooltip";
import { fmtPercent } from "../../domain/format";
import { useExperimentsStore } from "../../stores/experiments";
import Icon from "./Icon.vue";
import PulseDot from "./PulseDot.vue";

const props = defineProps<{ id: string }>();
const emit = defineEmits<{ open: [id: string] }>();
const experiments = useExperimentsStore();

const info = computed(() => experiments.info(props.id));
const running = computed(() => info.value?.status === "RUNNING");
const health = computed(() => experiments.health(props.id));
const colour = computed(() => experiments.colours[props.id]);
const tip = computed(() => {
  const d = experiments.detail(props.id);
  const parts = [props.id];
  if (d) parts.push(`${d.algo ?? "?"} · ${d.n_runs} runs`);
  if (running.value) parts.push(`running ${fmtPercent(info.value?.progress)}`);
  const h = experiments.healthText(props.id);
  if (h && h !== "Healthy") parts.push(h);
  return parts.join(" · ") + " — click for details";
});
</script>

<template>
  <span class="xpill" :class="{ running, missing: health === 'error' && !experiments.detail(id) }" role="button" tabindex="0" v-tip="tip" @click="emit('open', id)" @keydown.enter="emit('open', id)">
    <PulseDot v-if="running" :color="colour" :size="9" />
    <span v-else class="dot" :style="{ '--c': colour }" />
    <span class="nm">{{ experiments.name(id) }}</span>
    <span v-if="running" class="pct">{{ fmtPercent(info?.progress) }}</span>
    <span v-if="health === 'warning'" class="hicon warning" aria-label="warnings">⚠</span>
    <span v-else-if="health === 'error'" class="hicon error" aria-label="errors">✖</span>
    <button class="x" :aria-label="`Unload ${id}`" v-tip="'Unload'" @click.stop="experiments.unload(id)"><Icon name="x" :size="12" /></button>
    <span v-if="running" class="pbar" aria-hidden="true"><i :style="{ width: Math.round((info?.progress ?? 0) * 100) + '%' }" /></span>
  </span>
</template>

<style scoped>
.xpill {
  position: relative;
  overflow: hidden;
  cursor: pointer;
  transition:
    transform 0.15s,
    box-shadow 0.15s;
}
.xpill:hover {
  transform: translateY(-1px);
  box-shadow: var(--sh-hi);
}
.xpill.missing .nm {
  text-decoration: line-through;
  color: var(--ink3);
}
.nm {
  max-width: 190px;
  overflow: hidden;
  text-overflow: ellipsis;
}
.pct {
  font-size: 10.5px;
  color: var(--ink3);
  font-weight: 500;
}
.hicon {
  font-size: 11px;
}
.hicon.warning {
  color: var(--warn);
}
.hicon.error {
  color: var(--err);
}
.x {
  border: 0;
  background: none;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  color: var(--ink3);
  display: grid;
  place-items: center;
  padding: 0;
}
.x:hover {
  background: var(--err-soft);
  color: var(--err);
}
.pbar {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: 2px;
  background: var(--line);
}
.pbar i {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, var(--acc), color-mix(in srgb, var(--acc) 50%, #fff), var(--acc));
  background-size: 200% 100%;
  animation: shimmer 2s linear infinite;
  transition: width 0.4s ease;
}
@media (prefers-reduced-motion: reduce) {
  .pbar i {
    animation: none;
    background: var(--acc);
  }
}
</style>
