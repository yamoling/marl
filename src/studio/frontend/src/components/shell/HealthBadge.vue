<script setup lang="ts">
/** Health badge: "✓ healthy", "⚠ 2 warnings" or "✖ 1 error · 1 more". */
import { computed } from "vue";
import type { Health } from "../../api";
import { vTip } from "../../composables/tooltip";

const props = defineProps<{ health: Health; counts: { info: number; warning: number; error: number }; tip?: string }>();
const text = computed(() => {
  const { error, warning, info } = props.counts;
  const plural = (n: number, w: string) => `${n} ${w}${n > 1 ? "s" : ""}`;
  if (props.health === "error") return `✖ ${plural(Math.max(1, error), "error")}${warning + info ? ` · ${warning + info} more` : ""}`;
  if (props.health === "warning") return `⚠ ${plural(Math.max(1, warning), "warning")}`;
  return "✓ healthy";
});
</script>

<template>
  <span v-tip="tip" class="hb" :class="health">{{ text }}</span>
</template>
