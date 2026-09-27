<script setup lang="ts">
/**
 * Action visualiser (port of the old `action/ActionPanel.vue`): agent filter chips, then the
 * discrete action matrix or the continuous action radar depending on the action space.
 */
import { computed, ref, watch } from "vue";
import type { ReplayEpisode } from "../../api";
import { isDiscreteSpace, nAgents } from "../../domain/replay";
import ContinuousActionRadar from "./ContinuousActionRadar.vue";
import DiscreteActionMatrix from "./DiscreteActionMatrix.vue";

const props = defineProps<{ replay: ReplayEpisode; t: number }>();

const n = computed(() => nAgents(props.replay));
const selectedAgents = ref<number[]>([]);
watch(n, (v) => (selectedAgents.value = Array.from({ length: v }, (_, i) => i)), { immediate: true });
/** Actions exist for t < length: the terminal frame shows the last action. */
const safeT = computed(() => Math.max(0, Math.min(props.replay.episode.actions.length - 1, props.t)));
const discrete = computed(() => isDiscreteSpace(props.replay.action_space));

/** Toggle an agent, keeping at least one selected. @ai-generated */
function toggle(agent: number): void {
  const s = new Set(selectedAgents.value);
  if (s.has(agent)) {
    if (s.size === 1) return;
    s.delete(agent);
  } else s.add(agent);
  selectedAgents.value = [...s].sort((a, b) => a - b);
}
</script>

<template>
  <section class="apanel" aria-label="Actions">
    <header>
      <span class="lbl">Actions</span>
      <span class="kind">{{ replay.action_space ? (discrete ? "discrete" : "continuous") : "unknown space" }}</span>
      <span v-if="t > safeT" class="muted small">terminal state: last action shown</span>
    </header>
    <div v-if="n > 1" class="agents" role="group" aria-label="Agents shown">
      <button v-for="a in n" :key="a" type="button" class="achip" :class="{ on: selectedAgents.includes(a - 1) }" :aria-pressed="selectedAgents.includes(a - 1)" @click="toggle(a - 1)">
        A{{ a - 1 }}
      </button>
    </div>
    <p v-if="!replay.action_space" class="muted small">The replay has no action space description.</p>
    <p v-else-if="!replay.episode.actions.length" class="muted small">The replay has no actions.</p>
    <DiscreteActionMatrix v-else-if="discrete" :replay="replay" :t="safeT" :agents="selectedAgents" />
    <ContinuousActionRadar v-else :replay="replay" :t="safeT" :agents="selectedAgents" />
  </section>
</template>

<style scoped>
.apanel {
  border: 1px solid var(--line);
  border-radius: var(--r-sm);
  background: var(--card);
  padding: 8px 10px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 0;
}
header {
  display: flex;
  align-items: center;
  gap: 8px;
}
.lbl {
  font-size: var(--fs-eyebrow);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink3);
  font-weight: 600;
}
.kind {
  font-size: 10.5px;
  font-weight: 600;
  padding: 0 7px;
  border-radius: 6px;
  background: var(--seg-bg);
  color: var(--ink2);
  text-transform: uppercase;
}
.small {
  font-size: var(--fs-xs);
  margin: 0;
}
.agents {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.achip {
  border: 1px solid var(--line2);
  background: var(--card);
  border-radius: 999px;
  padding: 0 9px;
  font-size: var(--fs-xs);
}
.achip.on {
  border-color: var(--acc);
  background: var(--acc-soft);
  color: var(--acc2);
  font-weight: 700;
}
</style>
