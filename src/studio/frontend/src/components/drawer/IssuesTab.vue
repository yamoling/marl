<script setup lang="ts">
/** Issues tab: capability summary (lazy health check) and every issue with expandable detail. */
import type { ExperimentDetail } from "../../api";
import { useExperimentsStore } from "../../stores/experiments";
import CapabilityList from "./CapabilityList.vue";

const props = defineProps<{ detail: ExperimentDetail }>();
const experiments = useExperimentsStore();
const ICON = { error: "✖", warning: "⚠", info: "ℹ" } as const;
</script>

<template>
  <div>
    <h4>Capabilities</h4>
    <CapabilityList :capabilities="detail.capabilities" :issues="detail.issues" :checking="!!experiments.checking[detail.id]" @check="experiments.checkHealth(props.detail.id)" />
    <h4>Issues ({{ detail.issues.length }})</h4>
    <p v-if="!detail.issues.length" class="muted">No issue: this experiment is healthy.</p>
    <ul class="issues">
      <li v-for="(i, k) in detail.issues" :key="k" :class="i.level">
        <div class="ih">
          <span class="lvl">{{ ICON[i.level] }} {{ i.level }}</span>
          <code class="code">{{ i.code }}</code>
          <span v-if="i.scope" class="muted scope">{{ i.scope }}</span>
          <code v-if="i.path" class="path">{{ i.path }}</code>
        </div>
        <div class="msg">{{ i.message }}</div>
        <details v-if="i.detail">
          <summary>Detail</summary>
          <pre>{{ i.detail }}</pre>
        </details>
      </li>
    </ul>
  </div>
</template>

<style scoped>
h4 {
  font-size: var(--fs-eyebrow);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink3);
  margin: 6px 0 8px;
}
.issues {
  list-style: none;
  padding: 0;
  margin: 0;
  display: grid;
  gap: 8px;
}
li {
  border: 1px solid var(--line);
  border-left: 3px solid var(--ink3);
  border-radius: 8px;
  padding: 8px 10px;
}
li.error {
  border-left-color: var(--err);
}
li.warning {
  border-left-color: var(--warn);
}
.ih {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
  font-size: 11.5px;
}
.lvl {
  font-weight: 700;
  text-transform: uppercase;
  font-size: 10.5px;
}
.error .lvl {
  color: var(--err);
}
.warning .lvl {
  color: var(--warn);
}
.code,
.path {
  font-size: 11px;
  background: #f3f3f7;
  border-radius: 4px;
  padding: 0 5px;
}
.msg {
  margin-top: 4px;
}
pre {
  font-size: 11px;
  background: #fafafc;
  border-radius: 6px;
  padding: 8px;
  white-space: pre-wrap;
  margin: 4px 0 0;
}
summary {
  cursor: pointer;
  color: var(--ink3);
  font-size: 11.5px;
  margin-top: 4px;
}
</style>
