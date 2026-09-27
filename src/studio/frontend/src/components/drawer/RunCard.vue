<script setup lang="ts">
/**
 * Run card: seed, status (pulse when running), progress, issues, Stop / Restart actions and, in
 * the Runs tab, a small multiple of the default metric (the run's line over the faint mean).
 */
import { computed } from "vue";
import type { Issue } from "../../api";
import { buildPath } from "../../charts/render";
import { vTip } from "../../composables/tooltip";
import { useRunActions } from "../../composables/useRunActions";
import { fmtPercent, fmtStep } from "../../domain/format";
import type { LiveRun } from "../../stores/experiments";
import Icon from "../shell/Icon.vue";
import ProgressBar from "../shell/ProgressBar.vue";
import PulseDot from "../shell/PulseDot.vue";

type Line = { x: number[]; y: (number | null)[] };
const props = withDefaults(
  defineProps<{
    experimentId: string;
    run: LiveRun;
    issues?: Issue[];
    launchable: boolean | null;
    colour?: string;
    line?: Line | null;
    mean?: Line | null;
    domain?: { x: [number, number]; y: [number, number] } | null;
    clickable?: boolean;
  }>(),
  { issues: () => [], colour: "#6d5ef5", line: null, mean: null, domain: null, clickable: false },
);
const emit = defineEmits<{ open: [] }>();
const actions = useRunActions();

const W = 200;
const H = 44;
const paths = computed(() => {
  const d = props.domain;
  if (!d) return null;
  const X = (v: number) => 1 + ((v - d.x[0]) / (d.x[1] - d.x[0] || 1)) * (W - 2);
  const Y = (v: number) => H - 2 - ((v - d.y[0]) / (d.y[1] - d.y[0] || 1)) * (H - 4);
  return { mean: props.mean ? buildPath(props.mean.x, props.mean.y, X, Y) : "", run: props.line ? buildPath(props.line.x, props.line.y, X, Y) : "" };
});
const running = computed(() => props.run.status === "RUNNING");
const canRestart = computed(() => (props.run.status === "CANCELLED" || props.run.status === "CREATED") && props.launchable === true);
const restartTip = computed(() =>
  props.launchable === false ? "Cannot restart: the experiment cannot be launched (see Issues)" : props.launchable === null ? "Checking whether the experiment can be launched…" : "Restart this run",
);
const issueTip = computed(() => props.issues.map((i) => `${i.level}: ${i.message}`).join("\n"));
</script>

<template>
  <article class="runcard" :class="[run.status.toLowerCase(), { clickable }]" :data-run="run.id" :tabindex="clickable ? 0 : undefined" @click="clickable && emit('open')" @keydown.enter="clickable && emit('open')">
    <header>
      <b>{{ run.seed === null ? run.dirname : `seed ${run.seed}` }}</b>
      <span class="st"><PulseDot v-if="running" :size="6" />{{ run.status.toLowerCase() }}<template v-if="running"> {{ fmtPercent(run.progress) }}</template></span>
      <span v-if="issues.length" class="iss" v-tip="issueTip">⚠ {{ issues.length }}</span>
      <span class="sp" />
      <button v-if="running" type="button" class="hbtn danger" v-tip="'Stop this run'" :aria-label="`Stop ${run.dirname}`" @click.stop="actions.stopRun(experimentId, run.id)"><Icon name="stop" :size="13" /></button>
      <span v-else-if="run.status === 'CANCELLED' || run.status === 'CREATED'" v-tip="restartTip">
        <button type="button" class="hbtn" :disabled="!canRestart" :aria-label="`Restart ${run.dirname}`" @click.stop="actions.restartRun(experimentId, run.id)"><Icon name="restart" :size="13" /></button>
      </span>
    </header>
    <svg v-if="paths" class="spark" :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="none" aria-hidden="true">
      <path :d="paths.mean" fill="none" stroke="#b9b9c4" stroke-width="1.2" />
      <path :d="paths.run" fill="none" :stroke="colour" stroke-width="1.6" />
    </svg>
    <ProgressBar :value="run.progress" :running="running" :height="4" />
    <div class="meta muted">{{ run.dirname }} · step {{ fmtStep(run.latest_step) }}</div>
  </article>
</template>

<style scoped>
.runcard {
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 8px 10px;
  display: flex;
  flex-direction: column;
  gap: 5px;
  background: var(--card);
  transition: 0.15s;
}
.runcard.clickable {
  cursor: pointer;
}
.runcard.clickable:hover {
  border-color: #d4d0f3;
  box-shadow: var(--sh);
}
header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12.5px;
}
.st {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: var(--ink2);
}
.running .st {
  color: var(--acc2);
  font-weight: 600;
}
.cancelled .st,
.unknown .st {
  color: var(--warn);
}
.iss {
  font-size: 11px;
  color: var(--warn);
}
.sp {
  flex: 1;
}
.spark {
  width: 100%;
  height: 44px;
}
.meta {
  font-size: 10.5px;
}
</style>
