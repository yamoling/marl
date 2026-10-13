<script setup lang="ts">
/** Experiments shelf: toggle chips of the loaded experiments; all on = "all loaded" (includes future loads). */
import { computed } from "vue";
import { vTip } from "../../composables/tooltip";

const props = defineProps<{ experiments: "all" | string[]; loaded: { id: string; name: string; colour: string }[] }>();
const emit = defineEmits<{ "update:experiments": [v: "all" | string[]] }>();

const on = computed(() => (props.experiments === "all" ? new Set(props.loaded.map((e) => e.id)) : new Set(props.experiments)));

/**
 * Toggle one experiment. Selecting every loaded experiment goes back to "all"; ids referenced by
 * the plot but not loaded are kept so the plot can show them as unloaded.
 *
 * @ai-generated
 */
function toggle(id: string): void {
  const cur = on.value;
  const loadedIds = props.loaded.map((e) => e.id);
  const next = loadedIds.filter((x) => (x === id ? !cur.has(x) : cur.has(x)));
  if (next.length === loadedIds.length) return emit("update:experiments", "all");
  const unloadedRefs = props.experiments === "all" ? [] : props.experiments.filter((x) => !loadedIds.includes(x));
  emit("update:experiments", [...next, ...unloadedRefs]);
}
</script>

<template>
  <div class="shelf" data-shelf="experiments">
    <label>Experiments</label>
    <div class="shelf-body">
      <button
        v-for="e in loaded"
        :key="e.id"
        type="button"
        class="chip"
        :class="{ off: !on.has(e.id) }"
        :aria-pressed="on.has(e.id)"
        v-tip="e.id"
        @click="toggle(e.id)"
      >
        <span class="dot" :style="{ '--c': e.colour }" />{{ e.name }}
      </button>
      <span v-if="experiments === 'all'" class="muted all">all loaded</span>
      <span v-if="!loaded.length" class="muted all">none loaded</span>
    </div>
  </div>
</template>

<style scoped>
.all {
  font-size: 11px;
  padding: 0 4px;
}
</style>
