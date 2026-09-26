<script setup lang="ts">
/**
 * Start new runs of an experiment (product-spec §9): defaults from `launch-defaults`, client-side
 * validation (ranges and seed collisions), a preview line, the device picker, and server errors
 * shown inline (409 seed collision / not launchable, 400 validation, 502 early failure).
 */
import { computed, ref, watch } from "vue";
import { ApiError, type LaunchDefaults } from "../../api";
import { ESC_PRIORITY, useEscClose } from "../../composables/useEscStack";
import { errorText } from "../../composables/useRunActions";
import { launchDisabledReason } from "../../domain/capabilities";
import { formFromDefaults, previewLine, validateLaunch, type LaunchForm } from "../../domain/launch";
import { useExperimentsStore } from "../../stores/experiments";
import { useToasts } from "../../stores/toasts";
import { useUiStore } from "../../stores/ui";
import Icon from "../shell/Icon.vue";
import DevicePicker from "./DevicePicker.vue";

const ui = useUiStore();
const experiments = useExperimentsStore();
const toasts = useToasts();

const id = computed(() => ui.launchFor);
const defaults = ref<LaunchDefaults | null>(null);
const form = ref<LaunchForm | null>(null);
const loading = ref(false);
const loadError = ref("");
const submitting = ref(false);
const serverError = ref<{ field: keyof LaunchForm | null; message: string } | null>(null);
const touched = ref(false);

const close = () => (ui.launchFor = null);
useEscClose(() => !!id.value, close, ESC_PRIORITY.dialog);

/** Load defaults whenever the dialog opens for an experiment. @ai-generated */
async function loadDefaults(target: string): Promise<void> {
  loading.value = true;
  loadError.value = "";
  defaults.value = null;
  form.value = null;
  serverError.value = null;
  touched.value = false;
  try {
    const d = await experiments.launchDefaults(target);
    if (ui.launchFor !== target) return;
    defaults.value = d;
    form.value = formFromDefaults(d);
  } catch (e) {
    loadError.value = errorText(e);
  } finally {
    loading.value = false;
  }
}
watch(id, (v) => v && void loadDefaults(v), { immediate: true });

const validation = computed(() => (form.value && defaults.value ? validateLaunch(form.value, defaults.value.existing_seeds) : null));
const blocked = computed(() => (defaults.value ? launchDisabledReason(defaults.value.capabilities, defaults.value.issues) : null));
const preview = computed(() => (id.value && validation.value ? previewLine(id.value, validation.value.seeds) : ""));
const errorFor = (k: keyof LaunchForm) => validation.value?.errors[k] ?? (serverError.value?.field === k ? serverError.value.message : "");
const canSubmit = computed(() => !!validation.value?.ok && !blocked.value && !submitting.value);

/** Submit; server errors stay in the dialog. @ai-generated */
async function submit(): Promise<void> {
  touched.value = true;
  if (!id.value || !form.value || !canSubmit.value) return;
  submitting.value = true;
  serverError.value = null;
  const target = id.value;
  try {
    const runs = await experiments.startRuns(target, { ...form.value });
    toasts.push({ level: "success", message: `Started ${runs.length} run${runs.length === 1 ? "" : "s"} in ${experiments.name(target)}: ${runs.map((r) => r.split("/").pop()).join(", ")}` });
    close();
  } catch (e) {
    const seedClash = e instanceof ApiError && e.status === 409 && /seed/i.test(`${e.code} ${e.message}`);
    serverError.value = { field: seedClash ? "seed" : null, message: errorText(e) };
    if (seedClash) void experiments.launchDefaults(target).then((d) => (defaults.value = d));
  } finally {
    submitting.value = false;
  }
}

const num = (k: keyof LaunchForm, ev: Event) => {
  if (!form.value) return;
  const raw = (ev.target as HTMLInputElement).value;
  (form.value as Record<string, unknown>)[k] = raw === "" ? NaN : Number(raw);
  if (serverError.value?.field === k) serverError.value = null;
};
</script>

<template>
  <Teleport to="body">
    <div v-if="id" class="dlg-scrim" @click.self="close">
      <form class="dlg" role="dialog" aria-modal="true" aria-label="Start runs" novalidate @submit.prevent="submit">
        <header>
          <div>
            <div class="eyebrow">Start runs</div>
            <h3 class="mono">{{ id }}</h3>
          </div>
          <button type="button" class="close" aria-label="Close" @click="close"><Icon name="x" /></button>
        </header>

        <div v-if="loading" class="muted pad">Loading defaults and checking the experiment…</div>
        <div v-else-if="loadError" class="err pad">{{ loadError }}</div>
        <template v-else-if="form && defaults">
          <div v-if="blocked" class="err" role="alert">{{ blocked }}</div>
          <div class="grid">
            <label>
              <span>Number of runs</span>
              <input type="number" min="1" :value="form.n_runs" data-field="n_runs" @input="num('n_runs', $event)" />
              <small v-if="errorFor('n_runs')" class="fe">{{ errorFor("n_runs") }}</small>
            </label>
            <label>
              <span>First seed</span>
              <input type="number" min="0" :value="form.seed" data-field="seed" @input="num('seed', $event)" />
              <small v-if="errorFor('seed')" class="fe" role="alert">{{ errorFor("seed") }}</small>
              <small v-else class="muted">existing: {{ defaults.existing_seeds.length ? defaults.existing_seeds.join(", ") : "none" }}</small>
            </label>
            <label>
              <span>Tests per evaluation</span>
              <input type="number" min="1" :value="form.n_tests" data-field="n_tests" @input="num('n_tests', $event)" />
              <small v-if="errorFor('n_tests')" class="fe">{{ errorFor("n_tests") }}</small>
            </label>
            <label>
              <span>Test interval</span>
              <input type="number" min="1" step="1000" :value="form.test_interval" data-field="test_interval" @input="num('test_interval', $event)" />
              <small v-if="errorFor('test_interval')" class="fe">{{ errorFor("test_interval") }}</small>
            </label>
            <label>
              <span>Parallel jobs</span>
              <input type="number" min="1" :value="form.n_jobs" data-field="n_jobs" @input="num('n_jobs', $event)" />
              <small v-if="errorFor('n_jobs')" class="fe">{{ errorFor("n_jobs") }}</small>
            </label>
            <div class="checks">
              <label class="ck"><input v-model="form.save_weights" type="checkbox" /> Save weights</label>
              <label class="ck"><input v-model="form.save_actions" type="checkbox" /> Save actions (for replay)</label>
            </div>
          </div>
          <h4>Devices</h4>
          <DevicePicker v-model:device="form.device" v-model:disabled-devices="form.disabled_devices" v-model:gpu-strategy="form.gpu_strategy" />
          <p class="preview" data-role="preview">{{ preview }}</p>
          <div v-if="serverError && !serverError.field" class="err" role="alert">{{ serverError.message }}</div>
        </template>

        <footer>
          <button type="button" class="btn" @click="close">Cancel</button>
          <button type="submit" class="btn primary" :disabled="!canSubmit">
            <Icon name="play" :size="12" />{{ submitting ? "Starting…" : `Start ${validation?.seeds.length ?? ""} run${validation?.seeds.length === 1 ? "" : "s"}` }}
          </button>
        </footer>
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
  padding: 18px 22px 16px;
  width: min(640px, 94vw);
  max-height: 90vh;
  overflow: auto;
  animation: popin 0.12s ease-out;
}
header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 12px;
}
h3 {
  margin: 2px 0 0;
  font-size: 15px;
  word-break: break-all;
}
h4 {
  font-size: var(--fs-eyebrow);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink3);
  margin: 16px 0 8px;
}
.grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}
.grid label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
  color: var(--ink2);
}
.grid input[type="number"] {
  border: 1px solid var(--line2);
  border-radius: 8px;
  padding: 5px 9px;
  font-size: 13px;
}
.grid input:focus {
  outline: none;
  border-color: var(--acc);
  box-shadow: var(--focus-ring);
}
small {
  font-size: 11px;
}
.fe {
  color: var(--err);
}
.checks {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 6px;
}
.ck {
  flex-direction: row !important;
  align-items: center;
}
.ck input {
  accent-color: var(--acc);
}
.preview {
  margin: 14px 0 6px;
  font-family: var(--mono);
  font-size: 12px;
  color: var(--acc2);
  min-height: 1em;
}
.err {
  color: var(--err);
  background: var(--err-soft);
  border-radius: 8px;
  padding: 8px 10px;
  font-size: 12.5px;
  margin: 6px 0;
  white-space: pre-wrap;
}
.pad {
  padding: 16px 0;
}
footer {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 12px;
}
</style>
