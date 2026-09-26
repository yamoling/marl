<script setup lang="ts">
/**
 * Parameter tree: expandable object nodes, class names as tags, schedules as sparklines,
 * copy-path on hover. A search highlights matches and auto-expands their ancestors.
 */
import { computed, ref, watch } from "vue";
import type { ParamRow } from "../../api";
import { vTip } from "../../composables/tooltip";
import { defaultExpanded, visibleRows } from "../../domain/paramTree";
import { formatParamValue } from "../../domain/params";
import { copyText } from "../../stores/workspace";
import { useToasts } from "../../stores/toasts";
import Icon from "../shell/Icon.vue";
import ScheduleSpark from "./ScheduleSpark.vue";

const props = withDefaults(defineProps<{ rows: ParamRow[]; query?: string }>(), { query: "" });
const toasts = useToasts();
const expanded = ref<Set<string>>(defaultExpanded(props.rows));
watch(
  () => props.rows,
  (r) => (expanded.value = defaultExpanded(r)),
);
const visible = computed(() => visibleRows(props.rows, expanded.value, props.query));

function toggle(path: string): void {
  const s = new Set(expanded.value);
  if (s.has(path)) s.delete(path);
  else s.add(path);
  expanded.value = s;
}
function expandAll(open: boolean): void {
  expanded.value = open ? new Set(props.rows.filter((r) => r.kind === "object" || r.kind === "schedule").map((r) => r.path)) : new Set();
}
defineExpose({ expandAll });

/** Split `text` around case-insensitive occurrences of the query, for highlighting. @ai-generated */
function parts(text: string): { t: string; hit: boolean }[] {
  const q = props.query.trim().toLowerCase();
  if (!q) return [{ t: text, hit: false }];
  const out: { t: string; hit: boolean }[] = [];
  let i = 0;
  const low = text.toLowerCase();
  for (let k = low.indexOf(q); k >= 0; k = low.indexOf(q, i)) {
    if (k > i) out.push({ t: text.slice(i, k), hit: false });
    out.push({ t: text.slice(k, k + q.length), hit: true });
    i = k + q.length;
  }
  if (i < text.length) out.push({ t: text.slice(i), hit: false });
  return out;
}
const display = (r: ParamRow) => (r.value === null ? "null" : typeof r.value === "string" ? `"${r.value}"` : formatParamValue(r.value));

async function copy(path: string): Promise<void> {
  await copyText(path);
  toasts.push({ message: `Copied ${path}` });
}
</script>

<template>
  <div class="tree" role="tree">
    <div
      v-for="v in visible"
      :key="v.row.path"
      class="row"
      :class="{ match: v.match, node: v.hasChildren }"
      role="treeitem"
      :aria-expanded="v.hasChildren ? v.expanded : undefined"
      :data-path="v.row.path"
      :style="{ paddingLeft: 6 + v.row.depth * 16 + 'px' }"
    >
      <button v-if="v.hasChildren" type="button" class="tw" :aria-label="v.expanded ? 'Collapse' : 'Expand'" @click="toggle(v.row.path)">
        <Icon :name="v.expanded ? 'chevron-down' : 'chevron-right'" :size="12" />
      </button>
      <span v-else class="tw" />
      <span class="key"><template v-for="(p, i) in parts(v.row.key)" :key="i"><mark v-if="p.hit">{{ p.t }}</mark><template v-else>{{ p.t }}</template></template></span>
      <span v-if="v.row.cls" class="cls" :class="{ sch: v.row.kind === 'schedule' }"><template v-for="(p, i) in parts(v.row.cls)" :key="i"><mark v-if="p.hit">{{ p.t }}</mark><template v-else>{{ p.t }}</template></template></span>
      <ScheduleSpark v-if="v.row.kind === 'schedule'" :row="v.row" labels />
      <span v-if="v.row.kind !== 'object' && v.row.kind !== 'schedule'" class="val" :class="v.row.kind"
        ><template v-for="(p, i) in parts(display(v.row))" :key="i"><mark v-if="p.hit">{{ p.t }}</mark><template v-else>{{ p.t }}</template></template></span
      >
      <button type="button" class="cp" v-tip="`Copy path ${v.row.path}`" :aria-label="`Copy path ${v.row.path}`" @click="copy(v.row.path)"><Icon name="copy" :size="12" /></button>
    </div>
    <div v-if="!visible.length" class="muted empty">{{ query ? `No parameter matches “${query}”.` : "No parameters." }}</div>
  </div>
</template>

<style scoped>
.tree {
  font-size: 12.5px;
}
.row {
  display: flex;
  align-items: center;
  gap: 6px;
  min-height: 26px;
  border-radius: 6px;
  padding-right: 6px;
}
.row:hover {
  background: #f6f6fa;
}
.row.match {
  background: #fffbea;
}
.tw {
  width: 16px;
  height: 16px;
  border: 0;
  background: none;
  padding: 0;
  display: grid;
  place-items: center;
  color: var(--ink3);
  flex: none;
}
.key {
  color: var(--ink2);
}
.node > .key {
  font-weight: 600;
  color: var(--ink);
}
.cls {
  font-size: 10.5px;
  font-weight: 600;
  padding: 0 6px;
  border-radius: 5px;
  background: #f0f0f5;
  color: var(--ink2);
}
.cls.sch {
  background: var(--param-soft);
  color: var(--param-ink);
}
.val {
  font-family: var(--mono);
  font-size: 12px;
  color: var(--ink);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  min-width: 0;
}
.val.string {
  color: #8a4b08;
}
.val.boolean,
.val.null {
  color: var(--acc2);
}
.cp {
  margin-left: auto;
  border: 0;
  background: none;
  color: var(--ink3);
  opacity: 0;
  padding: 2px;
  border-radius: 4px;
}
.row:hover .cp,
.cp:focus-visible {
  opacity: 1;
}
.cp:hover {
  color: var(--acc2);
  background: var(--acc-soft);
}
mark {
  background: #ffe58a;
  color: inherit;
  border-radius: 2px;
}
.empty {
  padding: 14px 6px;
}
</style>
