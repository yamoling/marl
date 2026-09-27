<script setup lang="ts">
/**
 * Choose the timeline tracks of an experiment's replays (port of the old `modals/TrackWizard.vue`):
 * a draft selection with one card per track or track group (group toggle, per-group kind), applied
 * to the tracks store. Without a stored selection the draft starts with every track, kinds from the
 * settings' track rules.
 */
import { ref, watch } from "vue";
import { ESC_PRIORITY, useEscClose } from "../../composables/useEscStack";
import { resolveTrackKind, type TrackKind } from "../../domain/settings";
import { leafTracks, majorityKind, type TrackConfig, type TrackNode } from "../../domain/timeline";
import { useSettingsStore } from "../../stores/settings";
import { useTracksStore } from "../../stores/tracks";
import Icon from "../shell/Icon.vue";

const props = defineProps<{ open: boolean; experiment: string; tracks: TrackNode[] }>();
const emit = defineEmits<{ "update:open": [v: boolean] }>();
const store = useTracksStore();
const settings = useSettingsStore();
const draft = ref<TrackConfig[]>([]);
const close = () => emit("update:open", false);
useEscClose(() => props.open, close, ESC_PRIORITY.dialog);

/** Initialise the draft from the stored selection, else from every track (settings kinds). @ai-generated */
function init(): void {
  const stored = store.forExperiment(props.experiment);
  draft.value = stored.length
    ? stored.map((t) => ({ ...t }))
    : props.tracks.flatMap(leafTracks).map((t) => ({ label: t.label, kind: resolveTrackKind(settings.settings, t.label, t.kind) }));
}
watch(
  () => props.open,
  (o) => o && init(),
  { immediate: true },
);

const isOn = (label: string) => draft.value.some((d) => d.label === label);
const kindOf = (label: string, fallback: TrackKind = "numeric"): TrackKind => draft.value.find((d) => d.label === label)?.kind ?? fallback;
const groupOn = (n: TrackNode) => leafTracks(n).every((t) => isOn(t.label));
const groupAny = (n: TrackNode) => leafTracks(n).some((t) => isOn(t.label));
const groupKind = (n: TrackNode) => majorityKind(leafTracks(n).map((t) => kindOf(t.label, t.kind)));

/** Add or remove one leaf track from the draft. @ai-generated */
function toggle(label: string, kind: TrackKind, on: boolean): void {
  if (on && !isOn(label)) draft.value = [...draft.value, { label, kind: resolveTrackKind(settings.settings, label, kind) }];
  else if (!on) draft.value = draft.value.filter((d) => d.label !== label);
}
function toggleNode(n: TrackNode, on: boolean): void {
  for (const t of leafTracks(n)) toggle(t.label, t.kind, on);
}
function setKind(n: TrackNode, kind: TrackKind): void {
  const labels = new Set(leafTracks(n).map((t) => t.label));
  draft.value = draft.value.map((d) => (labels.has(d.label) ? { ...d, kind } : d));
}
function apply(): void {
  store.set(props.experiment, draft.value);
  close();
}
const checked = (ev: Event) => (ev.target as HTMLInputElement).checked;
const kindValue = (ev: Event) => (ev.target as HTMLSelectElement).value as TrackKind;
</script>

<template>
  <Teleport to="body">
    <div v-if="open" class="dlg-scrim" @click.self="close">
      <section class="dlg" role="dialog" aria-modal="true" aria-label="Timeline tracks">
        <header>
          <h3>Timeline tracks</h3>
          <button class="close" aria-label="Close" @click="close"><Icon name="x" /></button>
        </header>
        <div class="bar">
          <span class="muted">{{ draft.length }} selected</span>
          <button type="button" class="btn small" @click="draft = []"><Icon name="undo" :size="13" />Clear all</button>
        </div>
        <p v-if="!tracks.length" class="muted">This replay has no timeline data (rewards or agent details).</p>
        <div class="list">
          <section v-for="n in tracks" :key="n.label" class="opt">
            <div class="head">
              <label class="tog">
                <input type="checkbox" :checked="groupOn(n)" :indeterminate.prop="!groupOn(n) && groupAny(n)" @change="toggleNode(n, checked($event))" />
                <b :class="{ dim: !groupAny(n) }">{{ n.label }}</b>
                <span v-if="n.type === 'group'" class="muted">{{ n.subTracks.length }} tracks</span>
              </label>
              <select :disabled="!groupAny(n)" :value="groupKind(n)" :aria-label="`${n.label} representation`" @change="setKind(n, kindValue($event))">
                <option value="numeric">Numerical</option>
                <option value="categorical">Categorical</option>
              </select>
            </div>
            <div v-if="n.type === 'group'" class="subs">
              <label v-for="s in n.subTracks" :key="s.label" class="sub">
                <input type="checkbox" :checked="isOn(s.label)" @change="toggle(s.label, s.kind, checked($event))" />
                <span>{{ s.label }}</span>
              </label>
            </div>
          </section>
        </div>
        <footer>
          <button type="button" class="btn" @click="close">Cancel</button>
          <button type="button" class="btn primary" @click="apply">Apply tracks</button>
        </footer>
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
  padding: 18px 22px 16px;
  width: min(680px, 94vw);
  max-height: 86vh;
  display: flex;
  flex-direction: column;
  gap: 10px;
  animation: popin 0.12s ease-out;
}
header,
footer,
.bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
footer {
  justify-content: flex-end;
}
h3 {
  margin: 0;
  font-size: 17px;
}
.list {
  overflow: auto;
  display: grid;
  gap: 8px;
  min-height: 0;
}
.opt {
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 8px 10px;
  background: var(--panel);
}
.head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.tog {
  display: flex;
  align-items: center;
  gap: 7px;
  min-width: 0;
}
.dim {
  opacity: 0.55;
}
select {
  border: 1px solid var(--line2);
  border-radius: 7px;
  background: var(--card);
  padding: 2px 6px;
  font-size: var(--fs-sm);
}
.subs {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 2px 12px;
  margin-top: 6px;
  font-size: var(--fs-sm);
}
.sub {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}
</style>
