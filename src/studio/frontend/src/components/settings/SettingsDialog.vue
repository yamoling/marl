<script setup lang="ts">
/**
 * Settings: plot defaults (statistic, x axis), replay rules (global "only saved actions" and
 * per-trainer rules with glob/regex keys) and workspace export/import/reset.
 */
import { ref } from "vue";
import { ESC_PRIORITY, useEscClose } from "../../composables/useEscStack";
import { download } from "../../domain/export";
import { useSettingsStore } from "../../stores/settings";
import { useToasts } from "../../stores/toasts";
import { useWorkspaceStore } from "../../stores/workspace";
import Icon from "../shell/Icon.vue";
import Segmented from "../shell/Segmented.vue";

const props = defineProps<{ open: boolean }>();
const emit = defineEmits<{ "update:open": [value: boolean] }>();
const store = useSettingsStore();
const workspace = useWorkspaceStore();
const toasts = useToasts();
const close = () => emit("update:open", false);
useEscClose(() => props.open, close, ESC_PRIORITY.dialog);

const newKey = ref("");
const newValue = ref(true);
const file = ref<HTMLInputElement | null>(null);
const confirmReset = ref(false);

function addRule(): void {
  store.setTrainerRule(newKey.value, newValue.value);
  newKey.value = "";
}

function exportWorkspace(): void {
  download(`marl-studio-workspace-${new Date().toISOString().slice(0, 10)}.json`, workspace.exportJSON(), "application/json");
}

/** Import a workspace file; per-plot failures are reported by the workspace store. @ai-generated */
async function importWorkspace(ev: Event): Promise<void> {
  const f = (ev.target as HTMLInputElement).files?.[0];
  if (!f) return;
  const r = workspace.importJSON(await f.text());
  (ev.target as HTMLInputElement).value = "";
  toasts.push({ level: r.failures.length ? "warning" : "success", message: `Imported ${r.workspace.plots.length} plot${r.workspace.plots.length === 1 ? "" : "s"} and ${r.workspace.experiments.length} experiment${r.workspace.experiments.length === 1 ? "" : "s"}` });
}

function resetWorkspace(): void {
  if (!confirmReset.value) {
    confirmReset.value = true;
    return;
  }
  workspace.reset();
  confirmReset.value = false;
  toasts.push({ message: "Workspace cleared" });
}
</script>

<template>
  <Teleport to="body">
    <div v-if="open" class="dlg-scrim" @click.self="close">
      <section class="dlg" role="dialog" aria-modal="true" aria-label="Settings">
        <header>
          <h3>Settings</h3>
          <button class="close" aria-label="Close" @click="close"><Icon name="x" /></button>
        </header>

        <h4>New plots</h4>
        <div class="row">
          <span>Statistic</span>
          <Segmented v-model="store.settings.plots.center" :options="[{ value: 'mean', label: 'mean' }, { value: 'median', label: 'median' }]" label="Default centre" />
          <Segmented
            v-model="store.settings.plots.band"
            :options="[
              { value: 'ci95', label: 'ci95' },
              { value: 'std', label: 'std' },
              { value: 'minmax', label: 'min–max' },
              { value: 'none', label: 'none' },
            ]"
            label="Default band"
          />
        </div>
        <div class="row">
          <span>X axis</span>
          <Segmented v-model="store.settings.plots.xAxis" :options="[{ value: 'time_step', label: 'time step' }, { value: 'wall_time', label: 'wall time' }]" label="Default x axis" />
        </div>

        <h4>Replay</h4>
        <label class="row check"><input v-model="store.settings.replay.globalOnlySavedActions" type="checkbox" /> Replay only saved actions (all trainers)</label>
        <p class="muted small">Per-trainer rules override it. Keys match the trainer name exactly, as a glob (<code>QMix*</code>, <code>{VDN,QMix}</code>) or as <code>/regex/i</code>.</p>
        <div v-for="(v, k) in store.settings.replay.trainerRules" :key="k" class="rule">
          <code>{{ k }}</code>
          <label class="check"><input type="checkbox" :checked="v" @change="store.setTrainerRule(String(k), ($event.target as HTMLInputElement).checked)" /> only saved actions</label>
          <button class="hbtn danger" :aria-label="`Remove rule ${k}`" @click="store.removeTrainerRule(String(k))"><Icon name="trash" :size="14" /></button>
        </div>
        <form class="rule" @submit.prevent="addRule">
          <input v-model="newKey" placeholder="Trainer name, glob or /regex/" aria-label="Trainer rule key" />
          <label class="check"><input v-model="newValue" type="checkbox" /> only saved actions</label>
          <button class="btn small" :disabled="!newKey.trim()">Add rule</button>
        </form>

        <h4>Workspace</h4>
        <div class="row">
          <button class="btn" @click="exportWorkspace"><Icon name="download" :size="14" />Export JSON</button>
          <button class="btn" @click="file?.click()"><Icon name="upload" :size="14" />Import JSON</button>
          <input ref="file" type="file" accept="application/json,.json" hidden @change="importWorkspace" />
          <span class="sp" />
          <button class="btn" :class="{ danger: confirmReset }" @click="resetWorkspace">{{ confirmReset ? "Click again to clear" : "Clear workspace" }}</button>
        </div>
      </section>
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
  padding: 18px 22px 20px;
  width: min(560px, 94vw);
  max-height: 88vh;
  overflow: auto;
  animation: popin 0.12s ease-out;
}
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
h3 {
  margin: 0;
  font-size: 17px;
}
h4 {
  font-size: var(--fs-eyebrow);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink3);
  margin: 18px 0 8px;
}
.row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}
.row > span:first-child {
  width: 70px;
  color: var(--ink2);
}
.check {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.check input {
  accent-color: var(--acc);
}
.small {
  font-size: var(--fs-sm);
  margin: 0 0 8px;
}
.rule {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 6px;
}
.rule code,
.rule input:not([type]) {
  flex: 1;
}
.rule input:not([type]) {
  border: 1px solid var(--line2);
  border-radius: 8px;
  padding: 5px 9px;
}
.sp {
  flex: 1;
}
</style>
