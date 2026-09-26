<script setup lang="ts">
/** Overview tab: spec sheet (extractors), schedules as mini curves, and compact run cards. */
import { computed } from "vue";
import type { ExperimentDetail } from "../../api";
import { scheduleRows, specSheet } from "../../domain/specSheet";
import { useExperimentsStore } from "../../stores/experiments";
import RunCard from "./RunCard.vue";
import ScheduleSpark from "./ScheduleSpark.vue";

const props = defineProps<{ detail: ExperimentDetail }>();
const experiments = useExperimentsStore();
const fields = computed(() => specSheet(props.detail.algo, props.detail.params));
const schedules = computed(() => scheduleRows(props.detail.params));
const runs = computed(() => experiments.runs(props.detail.id));
const issuesOf = (runId: string) => props.detail.runs.find((r) => r.id === runId)?.issues ?? [];
</script>

<template>
  <div class="ov">
    <dl v-if="fields.length" class="spec">
      <div v-for="f in fields" :key="f.label" class="item" :title="f.path ?? undefined">
        <dt>{{ f.label }}</dt>
        <dd :class="{ mono: f.mono }">{{ f.value }}</dd>
      </div>
    </dl>
    <p v-else class="muted">No readable parameters (see Issues).</p>

    <template v-if="schedules.length">
      <h4>Schedules</h4>
      <div class="scheds">
        <div v-for="s in schedules" :key="s.path" class="sched">
          <span class="mono path">{{ s.path }}</span>
          <span class="cls">{{ s.cls }}</span>
          <ScheduleSpark :row="s" :width="150" :height="30" labels />
        </div>
      </div>
    </template>

    <h4>Runs</h4>
    <div class="runs">
      <RunCard v-for="r in runs" :key="r.id" :experiment-id="detail.id" :run="r" :issues="issuesOf(r.id)" :launchable="detail.capabilities.launch" />
    </div>
    <p v-if="!runs.length" class="muted">No runs yet.</p>
  </div>
</template>

<style scoped>
.spec {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
  gap: 10px 18px;
  margin: 4px 0 8px;
}
.item dt {
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--ink3);
  font-weight: 700;
}
.item dd {
  margin: 2px 0 0;
  font-size: 17px;
  font-weight: 600;
  letter-spacing: -0.01em;
  word-break: break-word;
}
.item dd.mono {
  font-size: 11.5px;
  font-weight: 500;
}
h4 {
  font-size: var(--fs-eyebrow);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink3);
  margin: 18px 0 8px;
}
.scheds {
  display: grid;
  gap: 6px;
}
.sched {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.path {
  font-size: 12px;
  min-width: 170px;
}
.cls {
  font-size: 10.5px;
  font-weight: 600;
  padding: 0 6px;
  border-radius: 5px;
  background: var(--param-soft);
  color: var(--param-ink);
}
.runs {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 8px;
}
</style>
