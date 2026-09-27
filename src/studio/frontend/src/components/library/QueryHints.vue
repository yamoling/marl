<script setup lang="ts">
/** Example query chips under the library search box. */
const emit = defineEmits<{ pick: [q: string] }>();
const HINTS = ["memory_size>60000", "mixer=qmix", "lr<1e-3", "name~vdn"];
</script>

<template>
  <div class="hints">
    Try
    <button v-for="h in HINTS" :key="h" class="hint" type="button" @click="emit('pick', h)">{{ h }}</button>
    <span class="muted">· combine terms with spaces (AND); operators = != &gt; &lt; &gt;= &lt;= ~</span>
  </div>
</template>

<style scoped>
.hints {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin: 8px 0 4px;
  align-items: center;
  font-size: 11.5px;
  color: var(--ink3);
}
.hint {
  border: 1px solid var(--line2);
  background: #fafafc;
  border-radius: 6px;
  padding: 1px 8px;
  font-family: var(--mono);
  font-size: 11.5px;
  color: var(--ink2);
}
.hint:hover {
  border-color: var(--acc);
  color: var(--acc2);
}
</style>
