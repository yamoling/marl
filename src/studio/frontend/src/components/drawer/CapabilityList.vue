<script setup lang="ts">
/** Capability summary: Metrics, Parameters, Replay, Launch — each ✓/✗/partial/checking with its reason. */
import { computed } from "vue";
import type { Capabilities, Issue } from "../../api";
import { capabilityLines } from "../../domain/capabilities";

const props = defineProps<{ capabilities: Capabilities; issues: Issue[]; checking?: boolean }>();
const emit = defineEmits<{ check: [] }>();
const lines = computed(() => capabilityLines(props.capabilities, props.issues, props.checking));
const MARK = { yes: "✓", no: "✗", partial: "◐", checking: "…" } as const;
</script>

<template>
  <ul class="caps">
    <li v-for="l in lines" :key="l.key" :class="l.state" :data-cap="l.key">
      <span class="mark" aria-hidden="true">{{ MARK[l.state] }}</span>
      <b>{{ l.label }}</b>
      <span class="val">{{ l.value }}</span>
      <button v-if="l.state === 'checking' && !checking" type="button" class="btn small" @click="emit('check')">Check now</button>
      <span v-if="l.reason" class="why">{{ l.reason }}</span>
    </li>
  </ul>
</template>

<style scoped>
.caps {
  list-style: none;
  padding: 0;
  margin: 0 0 14px;
  display: grid;
  gap: 6px;
}
li {
  display: grid;
  grid-template-columns: 18px 90px auto 1fr;
  align-items: baseline;
  gap: 4px 8px;
  padding: 7px 10px;
  border-radius: 9px;
  background: #fafafc;
  border: 1px solid var(--line);
}
.mark {
  font-weight: 700;
}
.yes .mark {
  color: var(--ok);
}
.no .mark {
  color: var(--err);
}
.partial .mark {
  color: var(--warn);
}
.checking .mark {
  color: var(--acc);
}
.val {
  color: var(--ink2);
}
.checking .val {
  animation: pulse 1.4s infinite;
}
.why {
  grid-column: 2 / -1;
  color: var(--ink2);
  font-size: 12px;
}
</style>
