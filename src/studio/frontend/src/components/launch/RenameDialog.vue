<script setup lang="ts">
/** Rename an experiment (moves its folder). Client-side id validation; server errors (409…) inline. */
import { computed, nextTick, ref, watch } from "vue";
import { ESC_PRIORITY, useEscClose } from "../../composables/useEscStack";
import { errorText } from "../../composables/useRunActions";
import { validateExperimentId } from "../../domain/launch";
import { useExperimentsStore } from "../../stores/experiments";
import { useLibraryStore } from "../../stores/library";
import { useToasts } from "../../stores/toasts";
import { useUiStore } from "../../stores/ui";

const ui = useUiStore();
const experiments = useExperimentsStore();
const library = useLibraryStore();
const toasts = useToasts();
const value = ref("");
const busy = ref(false);
const serverError = ref("");
const input = ref<HTMLInputElement | null>(null);
const id = computed(() => ui.renameFor);

const close = () => {
  if (!busy.value) ui.renameFor = null;
};
useEscClose(() => !!id.value, close, ESC_PRIORITY.dialog);

watch(id, async (v) => {
  if (!v) return;
  value.value = v;
  serverError.value = "";
  await nextTick();
  input.value?.focus();
  const cut = v.lastIndexOf("/") + 1;
  input.value?.setSelectionRange(cut, v.length);
});

const known = computed(() => [...library.items.map((e) => e.id), ...experiments.loaded]);
const error = computed(() => (id.value ? validateExperimentId(value.value, id.value, known.value) : null));

/** @ai-generated */
async function submit(): Promise<void> {
  if (!id.value || error.value || busy.value) return;
  busy.value = true;
  serverError.value = "";
  const from = id.value;
  try {
    const to = await experiments.rename(from, value.value.trim());
    if (ui.drawerId === from) ui.drawerId = to;
    ui.renameFor = null;
    toasts.push({ level: "success", message: `Renamed ${from} → ${to}` });
  } catch (e) {
    serverError.value = errorText(e);
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <Teleport to="body">
    <div v-if="id" class="dlg-scrim" @click.self="close">
      <form class="dlg" role="dialog" aria-modal="true" aria-label="Rename experiment" @submit.prevent="submit">
        <h3>Rename experiment</h3>
        <p class="muted">Moves the folder under the logs root and updates <code>logdir</code> / <code>rundir</code> in its JSON files. Plots follow the new name.</p>
        <label>
          <span>New id (relative to the logs folder)</span>
          <input ref="input" v-model="value" class="mono" spellcheck="false" autocomplete="off" aria-label="New experiment id" @input="serverError = ''" />
        </label>
        <p v-if="error && value.trim() !== id" class="fe" role="alert">{{ error }}</p>
        <p v-if="serverError" class="err" role="alert">{{ serverError }}</p>
        <div class="acts">
          <button type="button" class="btn" @click="close">Cancel</button>
          <button type="submit" class="btn primary" :disabled="!!error || busy">{{ busy ? "Renaming…" : "Rename" }}</button>
        </div>
      </form>
    </div>
  </Teleport>
</template>

<style scoped>
.dlg-scrim {
  position: fixed;
  inset: 0;
  z-index: var(--z-dialog);
  background: var(--scrim);
  display: grid;
  place-items: center;
}
.dlg {
  background: var(--card);
  border-radius: 14px;
  box-shadow: var(--sh-sheet);
  padding: 20px 22px 16px;
  width: min(520px, 92vw);
  animation: popin 0.12s ease-out;
}
h3 {
  margin: 0 0 6px;
  font-size: 16px;
}
p {
  font-size: 12.5px;
}
label {
  display: flex;
  flex-direction: column;
  gap: 5px;
  font-size: 12px;
  color: var(--ink2);
}
input {
  border: 1px solid var(--line2);
  border-radius: 8px;
  padding: 6px 10px;
  font-size: 13px;
}
input:focus {
  outline: none;
  border-color: var(--acc);
  box-shadow: var(--focus-ring);
}
.fe {
  color: var(--err);
  margin: 6px 0 0;
}
.err {
  color: var(--err);
  background: var(--err-soft);
  border-radius: 8px;
  padding: 6px 10px;
}
.acts {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 14px;
}
</style>
