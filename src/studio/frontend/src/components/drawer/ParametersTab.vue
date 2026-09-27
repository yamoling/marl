<script setup lang="ts">
/** Parameters tab: searchable tree or raw JSON; a banner explains degraded parameter sets. */
import { computed, ref } from "vue";
import type { ExperimentDetail } from "../../api";
import { reasonIssue } from "../../domain/capabilities";
import { copyText } from "../../stores/workspace";
import { useToasts } from "../../stores/toasts";
import Icon from "../shell/Icon.vue";
import Segmented from "../shell/Segmented.vue";
import ParamTree from "./ParamTree.vue";

const props = defineProps<{ detail: ExperimentDetail }>();
const toasts = useToasts();
const q = ref("");
const mode = ref<"tree" | "raw">(props.detail.capabilities.params === "none" && props.detail.params.length === 0 ? "raw" : "tree");
const tree = ref<InstanceType<typeof ParamTree> | null>(null);

const degraded = computed(() => {
  const p = props.detail.capabilities.params;
  if (p === "full") return null;
  const why = reasonIssue(props.detail.issues, "params");
  const what = { partial: "Parameters are partial", raw: "Parameters are shown raw (the trainer could not be deserialized)", none: "No parameters could be read" }[p];
  return `${what}.${why ? ` ${why.message}` : ""} The raw JSON is always available.`;
});
const raw = computed(() => JSON.stringify(props.detail.raw, null, 2));
async function copyRaw(): Promise<void> {
  await copyText(raw.value);
  toasts.push({ message: "Copied experiment.json" });
}
</script>

<template>
  <div class="ptab">
    <div v-if="degraded" class="banner" role="note">⚠ {{ degraded }}</div>
    <div class="bar">
      <label v-if="mode === 'tree'" class="search">
        <Icon name="search" :size="13" class="muted" />
        <input v-model="q" placeholder="Search paths and values" aria-label="Search parameters" spellcheck="false" />
      </label>
      <span v-else class="sp" />
      <template v-if="mode === 'tree' && !q">
        <button type="button" class="btn small" @click="tree?.expandAll(true)">Expand all</button>
        <button type="button" class="btn small" @click="tree?.expandAll(false)">Collapse</button>
      </template>
      <Segmented
        v-model="mode"
        :options="[
          { value: 'tree', label: 'Tree' },
          { value: 'raw', label: 'Raw JSON' },
        ]"
        label="Parameters view"
      />
    </div>
    <ParamTree v-if="mode === 'tree'" ref="tree" :rows="detail.params" :query="q" />
    <div v-else class="raw">
      <button type="button" class="btn small cp" @click="copyRaw"><Icon name="copy" :size="12" />Copy</button>
      <pre>{{ raw }}</pre>
    </div>
  </div>
</template>

<style scoped>
.banner {
  background: var(--warn-soft);
  color: var(--warn);
  border-radius: 9px;
  padding: 8px 12px;
  margin-bottom: 10px;
  font-size: 12.5px;
}
.bar {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  position: sticky;
  top: 0;
  background: var(--card);
  padding: 4px 0;
  z-index: 1;
}
.search {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 6px;
  border: 1px solid var(--line2);
  border-radius: 8px;
  padding: 4px 8px;
}
.search:focus-within {
  border-color: var(--acc);
  box-shadow: var(--focus-ring);
}
.search input {
  border: 0;
  outline: none;
  flex: 1;
  min-width: 0;
  background: transparent;
}
.sp {
  flex: 1;
}
.raw {
  position: relative;
}
.raw pre {
  background: #fafafc;
  border: 1px solid var(--line);
  border-radius: 9px;
  padding: 12px;
  font-size: 11.5px;
  overflow: auto;
  max-height: 70vh;
  margin: 0;
}
.cp {
  position: absolute;
  right: 8px;
  top: 8px;
}
</style>
