<template>
    <div ref="modal" class="modal fade" tabindex="-1">
        <div class="modal-dialog modal-dialog-centered modal-lg modal-dialog-scrollable">
            <div class="modal-content timeline-modal">
                <div class="modal-header">
                    <div>
                        <h5 class="modal-title mb-1">Configure timeline tracks</h5>
                        <div class="text-muted small">Choose the exact components to plot and optionally rename each
                            track.</div>
                    </div>
                    <button type="button" class="btn-close" @click="close" aria-label="Close"></button>
                </div>

                <div class="modal-body timeline-body">
                    <div v-if="availableOptions.length === 0" class="empty-state">
                        No optional action-detail keys were found for this episode.
                    </div>

                    <div v-else class="wizard-panel">
                        <div class="wizard-toolbar mb-3">
                            <button type="button" class="btn btn-sm btn-outline-secondary" @click="selectAll">
                                Select all
                            </button>
                            <button type="button" class="btn btn-sm btn-outline-secondary" @click="selectNone">
                                Clear all
                            </button>
                            <span class="text-muted small ms-auto">The selection is persisted per experiment.</span>
                        </div>

                        <div class="option-list">
                            <section v-for="option in availableOptions" :key="option.key" class="option-card">
                                <div class="option-card-head">
                                    <label class="track-toggle">
                                        <input class="form-check-input option-check" type="checkbox"
                                            :checked="isSelected(option.key)"
                                            @change="toggleTrack(option.key, $event)" />
                                        <span class="option-label">{{ option.label }}</span>
                                    </label>

                                    <select class="form-select form-select-sm option-kind"
                                        :disabled="!isSelected(option.key)" :value="getFamilyKind(option.key)"
                                        :aria-label="`${option.label} family representation`"
                                        @change="setFamilyKind(option.key, $event)">
                                        <option value="numeric">Numerical</option>
                                        <option value="categorical">Categorical</option>
                                    </select>

                                    <input class="form-control form-control-sm option-alias"
                                        :value="getAlias(option.key)" :disabled="!isSelected(option.key)"
                                        placeholder="Optional alias" @input="setAlias(option.key, $event)" />
                                </div>

                                <div class="component-grid" :class="{ disabled: !isSelected(option.key) }">
                                    <label v-for="component in option.components" :key="componentKey(component.path)"
                                        class="component-option">
                                        <input class="form-check-input" type="checkbox"
                                            :disabled="!isSelected(option.key)"
                                            :checked="isComponentSelected(option.key, component.path)"
                                            @change="toggleComponent(option.key, component.path, $event)" />
                                        <span>{{ component.label }}</span>
                                    </label>
                                </div>
                            </section>
                        </div>
                    </div>
                </div>

                <div class="modal-footer">
                    <button type="button" class="btn btn-outline-secondary" @click="close">Cancel</button>
                    <button type="button" class="btn btn-primary" @click="confirm">
                        Apply tracks
                    </button>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { Modal } from 'bootstrap';
import { ReplayEpisode } from '../../models/Episode';
import { type TimelineTrackKind } from '../../models/Timeline';
import {
    discoverReplayTrackOptions,
    ReplayTrackOption,
    ReplayTrackSelection,
} from '../visualisation/replayTimeline';

const props = defineProps<{
    episode: ReplayEpisode | null;
    nAgents: number;
    selectedTracks: ReplayTrackSelection[];
}>();

const emits = defineEmits<{
    (event: 'confirm', keys: ReplayTrackSelection[]): void;
}>();

const modal = ref({} as HTMLDivElement);
let modalInstance: Modal | null = null;
const draftSelections = ref<ReplayTrackSelection[]>([]);

const availableOptions = computed(() => discoverReplayTrackOptions(props.episode, props.nAgents));

function showModal() {
    draftSelections.value = sanitizeSelections(props.selectedTracks, availableOptions.value);

    if (modalInstance == null) {
        modalInstance = new Modal(modal.value);
    }

    modalInstance.show();
}

function confirm() {
    emits('confirm', draftSelections.value.slice());
    modalInstance?.hide();
}

function close() {
    modalInstance?.hide();
}

function selectAll() {
    draftSelections.value = availableOptions.value.map((option) => ({
        key: option.key,
        alias: getSelection(option.key)?.alias ?? null,
        componentPaths: option.components.map((component) => component.path),
        componentKinds: option.components.map((_, index) => getSelection(option.key)?.componentKinds[index] ?? ('numeric' as TimelineTrackKind)),
    }));
}

function selectNone() {
    draftSelections.value = [];
}

function isSelected(key: string): boolean {
    return draftSelections.value.some((selection) => selection.key === key);
}

function getSelection(key: string): ReplayTrackSelection | null {
    return draftSelections.value.find((selection) => selection.key === key) ?? null;
}

function getAlias(key: string): string {
    return getSelection(key)?.alias ?? '';
}

function getFamilyKind(key: string): TimelineTrackKind {
    const selection = getSelection(key);
    if (selection == null) return 'numeric';

    return selection.componentKinds[0] ?? 'numeric';
}

function toggleTrack(key: string, event: Event) {
    const target = event.target as HTMLInputElement;

    if (target.checked) {
        if (!isSelected(key)) {
            const option = availableOptions.value.find((entry) => entry.key === key);
            if (option == null) return;

            draftSelections.value = [...draftSelections.value, {
                key,
                alias: null,
                componentPaths: option.components.map((component) => component.path),
                componentKinds: option.components.map(() => 'numeric'),
            }];
        }
        return;
    }

    draftSelections.value = draftSelections.value.filter((selection) => selection.key !== key);
}

function toggleComponent(key: string, path: number[], event: Event) {
    const target = event.target as HTMLInputElement;
    const selection = getSelection(key);
    if (selection == null) return;

    const removedIndex = selection.componentPaths.findIndex((candidate) => arePathsEqual(candidate, path));
    let componentPaths = selection.componentPaths.filter((candidate) => !arePathsEqual(candidate, path));
    let componentKinds = selection.componentKinds.filter((_, index) => index !== removedIndex);
    if (target.checked) {
        componentPaths = [...componentPaths, path];
        componentKinds = [...componentKinds, getFamilyKind(key)];
    }

    if (componentPaths.length === 0) {
        draftSelections.value = draftSelections.value.filter((entry) => entry.key !== key);
        return;
    }

    draftSelections.value = draftSelections.value.map((entry) => entry.key === key
        ? { ...entry, componentPaths, componentKinds }
        : entry);
}

function setFamilyKind(key: string, event: Event) {
    const target = event.target as HTMLSelectElement;
    const kind = normalizeKind(target.value);

    draftSelections.value = draftSelections.value.map((entry) => entry.key === key
        ? { ...entry, componentKinds: entry.componentKinds.map(() => kind) }
        : entry);
}

function setAlias(key: string, event: Event) {
    const target = event.target as HTMLInputElement;
    draftSelections.value = draftSelections.value.map((entry) => entry.key === key
        ? { ...entry, alias: target.value.trim().length > 0 ? target.value : null }
        : entry);
}

function isComponentSelected(key: string, path: number[]): boolean {
    return getSelection(key)?.componentPaths.some((candidate) => arePathsEqual(candidate, path)) ?? false;
}

function componentKey(path: number[]): string {
    return path.length === 0 ? 'scalar' : path.join('.');
}

function sanitizeSelections(selections: ReplayTrackSelection[], options: ReplayTrackOption[]): ReplayTrackSelection[] {
    const optionByKey = new Map(options.map((option) => [option.key, option] as const));
    const nextSelections: ReplayTrackSelection[] = [];

    for (const selection of selections) {
        const option = optionByKey.get(selection.key);
        if (option == null) continue;

        const validEntries = selection.componentPaths
            .map((path, index) => ({ path, kind: normalizeKind(selection.componentKinds[index]) }))
            .filter((entry, index, entries) => entries.findIndex((candidate) => arePathsEqual(candidate.path, entry.path)) === index)
            .filter((entry) => option.components.some((component) => arePathsEqual(component.path, entry.path)));

        if (validEntries.length === 0) continue;

        nextSelections.push({
            key: selection.key,
            alias: selection.alias?.trim().length ? selection.alias.trim() : null,
            componentPaths: validEntries.map((entry) => entry.path),
            componentKinds: validEntries.map((entry) => entry.kind),
        });
    }

    return nextSelections;
}

function arePathsEqual(left: number[], right: number[]): boolean {
    if (left.length !== right.length) return false;
    return left.every((value, index) => value === right[index]);
}

function normalizeKind(kind: string): TimelineTrackKind {
    return kind === 'categorical' ? 'categorical' : 'numeric';
}

defineExpose({ showModal });
</script>

<style scoped>
.timeline-modal {
    border: 1px solid rgb(221, 211, 197);
    border-radius: 0.8rem;
    background: rgba(255, 255, 255, 0.98);
    box-shadow: 0 20px 40px rgba(15, 23, 42, 0.14);
}

.timeline-body {
    padding-top: 0.5rem;
    padding-bottom: 0.75rem;
}

.wizard-panel {
    display: grid;
    gap: 0.75rem;
}

.wizard-toolbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.45rem;
}

.option-list {
    display: grid;
    gap: 0.55rem;
}

.option-card {
    display: grid;
    gap: 0.55rem;
    padding: 0.8rem;
    border: 1px solid var(--bs-border-color);
    border-radius: 0.7rem;
    background: color-mix(in srgb, var(--bs-body-bg) 94%, var(--bs-primary) 6%);
}

.option-card-head {
    display: grid;
    gap: 0.5rem;
}

.track-toggle {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
}

.option-label {
    font-weight: 700;
}

.option-alias {
    max-width: 28rem;
}

.component-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr));
    gap: 0.35rem 0.75rem;
}

.component-grid.disabled {
    opacity: 0.55;
}

.component-option {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    min-width: 0;
}

.empty-state {
    padding: 0.9rem 1rem;
    border: 1px dashed var(--bs-border-color);
    border-radius: 0.7rem;
    color: var(--bs-secondary-color);
    background: color-mix(in srgb, var(--bs-body-bg) 96%, var(--bs-secondary) 4%);
}
</style>