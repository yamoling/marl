<template>
    <div ref="modal" class="modal fade" tabindex="-1" aria-labelledby="settings-modal-title" aria-hidden="true">
        <div class="modal-dialog modal-dialog-centered modal-xl modal-dialog-scrollable">
            <div class="modal-content settings-modal" style="height: 80vh">
                <div class="modal-header">
                    <div>
                        <h5 id="settings-modal-title" class="modal-title mb-1">Settings</h5>
                        <p class="settings-modal-subtitle mb-0 text-muted">
                            Homescreen and replay preferences are stored locally in this browser.
                        </p>
                    </div>
                    <button type="button" class="btn-close" @click="close" aria-label="Close"></button>
                </div>

                <div class="modal-body settings-layout">
                    <nav class="settings-nav" aria-label="Settings categories">
                        <button v-for="tab in tabs" :key="tab.id" type="button" class="settings-tab"
                            :class="{ active: activeTab === tab.id }" @click="activeTab = tab.id">
                            <span class="settings-tab-title">{{ tab.label }}</span>
                            <span v-if="tab.badge != null" class="badge text-bg-light settings-tab-badge">
                                {{ tab.badge }}
                            </span>
                        </button>
                    </nav>

                    <section class="settings-content">
                        <div v-if="activeTab === 'homescreen'" class="settings-section settings-panel">
                            <div class="settings-subsection">
                                <h6 class="settings-subsection-title mb-0">Plotting axis</h6>
                                <p class="text-muted mb-0">
                                    Plot results against wall-time instead of training steps. When enabled, granularity
                                    is interpreted
                                    as seconds.
                                </p>
                                <label class="form-check settings-switch">
                                    <input class="form-check-input" type="checkbox"
                                        v-model="draft.homescreen.useWallTime"
                                        @keydown.enter="($event.target as HTMLInputElement)!.blur()" />
                                    <span class="form-check-label">Use wall-time on homescreen plots</span>
                                </label>
                            </div>

                            <hr class="settings-divider" />

                            <div class="section-heading">
                                <div>
                                    <h6 class="mb-1">Colours</h6>
                                    <p class="text-muted mb-0">Manage saved experiment colours used on the homescreen
                                        plots and tables.</p>
                                </div>
                                <button type="button" class="btn btn-sm btn-outline-secondary"
                                    @click="addColourRule">Add colour</button>
                            </div>

                            <div v-if="draft.homescreen.colours.length === 0" class="empty-state">No saved experiment
                                colours yet.</div>

                            <div v-for="(rule, index) in draft.homescreen.colours" :key="`colour-${index}`"
                                class="rule-row rule-row--color">
                                <div class="color-preview" :style="{ backgroundColor: rule.value }"></div>
                                <input v-model="rule.key" type="text" class="form-control form-control-sm rule-key"
                                    placeholder="logs/experiment-name" />
                                <input v-model="rule.value" type="color"
                                    class="form-control form-control-color color-input" title="Edit colour" />
                                <input v-model="rule.value" type="text" class="form-control form-control-sm color-hex"
                                    placeholder="#rrggbb" />
                                <button type="button" class="btn btn-sm btn-outline-danger"
                                    @click="removeColourRule(index)">Remove</button>
                            </div>

                            <div class="settings-reset-actions">
                                <button type="button" class="btn btn-outline-danger" @click="clearColours">Clear saved
                                    colours</button>
                            </div>
                        </div>

                        <div v-else-if="activeTab === 'replay'" class="settings-section settings-panel">
                            <div class="settings-subsection">
                                <h6 class="settings-subsection-title mb-0">Timeline</h6>
                                <div class="section-heading">
                                    <div>
                                        <h6 class="mb-1">Track types</h6>
                                        <p class="text-muted mb-0">
                                            Rule keys support exact labels, glob-like patterns (for example,
                                            <code>{O,o}ptions*</code>), or explicit regex with slashes (for example,
                                            <code>/^options/i</code>).
                                        </p>
                                    </div>
                                </div>

                                <div class="rule-row rule-row--adding">
                                    <div class="rule-key-cell">
                                        <input v-model="newTimelineKindRule.key" type="text"
                                            class="form-control form-control-sm rule-key"
                                            placeholder="Track label or pattern"
                                            @keydown.enter.prevent="commitNewTimelineKindRule" />
                                        <span class="badge text-bg-secondary rule-match-badge"
                                            :title="matchTooltip(newTimelineKindRule.key)">
                                            {{ matchCount(newTimelineKindRule.key) }}
                                        </span>
                                    </div>
                                    <select v-model="newTimelineKindRule.kind"
                                        class="form-select form-select-sm rule-value"
                                        @keydown.enter.prevent="commitNewTimelineKindRule">
                                        <option value="numeric">Numerical</option>
                                        <option value="categorical">Categorical</option>
                                    </select>
                                    <button type="button" class="btn btn-sm btn-outline-secondary"
                                        @click="commitNewTimelineKindRule" title="Add track type rule"
                                        aria-label="Add track type rule">
                                        <font-awesome-icon :icon="['fas', 'plus']" />
                                    </button>
                                </div>

                                <div v-for="(rule, index) in draft.replay.timelineKinds" :key="`timeline-kind-${index}`"
                                    class="rule-row">
                                    <div class="rule-key-cell">
                                        <input v-model="rule.key" type="text"
                                            class="form-control form-control-sm rule-key"
                                            placeholder="Track label or pattern"
                                            @keydown.enter="($event.target as HTMLInputElement)!.blur()" />
                                        <span class="badge text-bg-secondary rule-match-badge"
                                            :title="matchTooltip(rule.key)">
                                            {{ matchCount(rule.key) }}
                                        </span>
                                    </div>
                                    <select v-model="rule.kind" class="form-select form-select-sm rule-value">
                                        <option value="numeric">Numerical</option>
                                        <option value="categorical">Categorical</option>
                                    </select>
                                    <button type="button" class="btn btn-sm btn-outline-danger"
                                        @click="removeTrackKindRule(index)" title="Remove track type rule"
                                        aria-label="Remove track type rule">
                                        <font-awesome-icon :icon="['fas', 'trash']" />
                                    </button>
                                </div>
                            </div>

                            <hr class="settings-divider" />

                            <div class="settings-subsection">
                                <h6 class="settings-subsection-title mb-0">Replay type</h6>

                                <div class="settings-subsection">
                                    <div class="section-heading">
                                        <h6 class="mb-1">Default replay type</h6>
                                    </div>
                                    <p class="text-muted mb-0">
                                        If checked, the actions saved on disk from the training are used to replay the
                                        episode. If
                                        unchecked, the saved weights are loaded at replay-time to provide more
                                        information in the UI.
                                    </p>
                                    <label class="form-check settings-switch">
                                        <input class="form-check-input" type="checkbox"
                                            v-model="draft.replay.globalOnlySavedActions"
                                            @keydown.enter="($event.target as HTMLInputElement)!.blur()" />
                                        <span class="form-check-label">Only replay stored actions</span>
                                    </label>
                                </div>

                                <hr class="settings-divider" />

                                <div class="settings-subsection">
                                    <div class="section-heading">
                                        <div>
                                            <h6 class="mb-1">Trainer-level replay rules</h6>
                                            <p class="text-muted mb-0">
                                                Override replay type for all experiments with a specific trainer. Rule
                                                keys support exact
                                                names, glob-like patterns (for example, <code>chat-gpt*</code>), or
                                                explicit regex with
                                                slashes (for example, <code>/^llm/i</code>).
                                            </p>
                                        </div>
                                    </div>

                                    <div class="rule-row rule-row--adding">
                                        <div class="rule-key-cell">
                                            <input v-model="newTrainerReplayRule.key" type="text"
                                                class="form-control form-control-sm rule-key"
                                                placeholder="Trainer name or pattern"
                                                @keydown.enter.prevent="commitNewTrainerReplayRule" />
                                            <span class="badge text-bg-secondary rule-match-badge"
                                                :title="trainerMatchTooltip(newTrainerReplayRule.key)">
                                                {{ trainerMatchCount(newTrainerReplayRule.key) }}
                                            </span>
                                        </div>
                                        <select v-model="newTrainerReplayRule.value"
                                            class="form-select form-select-sm rule-value"
                                            @keydown.enter.prevent="commitNewTrainerReplayRule">
                                            <option :value="false">Allow agent replay</option>
                                            <option :value="true">Stored actions only</option>
                                        </select>
                                        <button type="button" class="btn btn-sm btn-outline-secondary"
                                            @click="commitNewTrainerReplayRule" title="Add trainer replay rule"
                                            aria-label="Add trainer replay rule">
                                            <font-awesome-icon :icon="['fas', 'plus']" />
                                        </button>
                                    </div>

                                    <div v-for="(rule, index) in draft.replay.trainerRules" :key="`trainer-${index}`"
                                        class="rule-row">
                                        <div class="rule-key-cell">
                                            <input v-model="rule.key" type="text"
                                                class="form-control form-control-sm rule-key"
                                                placeholder="Trainer name or pattern"
                                                @keydown.enter="($event.target as HTMLInputElement)!.blur()" />
                                            <span class="badge text-bg-secondary rule-match-badge"
                                                :title="trainerMatchTooltip(rule.key)">
                                                {{ trainerMatchCount(rule.key) }}
                                            </span>
                                        </div>
                                        <select v-model="rule.value" class="form-select form-select-sm rule-value">
                                            <option :value="false">Allow agent replay</option>
                                            <option :value="true">Stored actions only</option>
                                        </select>
                                        <button type="button" class="btn btn-sm btn-outline-danger"
                                            @click="removeReplayTrainerRule(index)" title="Remove trainer replay rule"
                                            aria-label="Remove trainer replay rule">
                                            <font-awesome-icon :icon="['fas', 'trash']" />
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div v-else class="settings-section settings-section--danger settings-panel">
                            <div class="section-heading">
                                <div>
                                    <h6 class="mb-1">Reset</h6>
                                    <p class="text-muted mb-0">Restore all settings to defaults.</p>
                                </div>
                            </div>

                            <div class="settings-reset-actions">
                                <button type="button" class="btn btn-outline-danger" @click="resetAll">Reset all
                                    settings</button>
                            </div>
                        </div>
                    </section>
                </div>

                <div class="modal-footer d-flex justify-content-between align-items-center">
                    <span class="settings-footer-note text-muted" v-if="hasChanges">You have unsaved changes.</span>
                    <span v-else class="settings-footer-note text-muted">Saved locally in this browser.</span>
                    <div class="d-flex gap-2">
                        <button type="button" class="btn btn-outline-secondary" @click="close">Cancel</button>
                        <button type="button" class="btn btn-primary" @click="save">Save settings</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import { Modal } from "bootstrap";
import { useSettingsStore } from "../../stores/SettingsStore";
import { useTracksStore } from "../../stores/TracksStore";
import { useExperimentStore } from "../../stores/ExperimentStore";
import type { TimelineTrackKind } from "../../models/Timeline";
import { matchesTrackRuleKey } from "../../models/Settings";

type EditableReplayRule = {
    key: string;
    value: boolean;
};

type EditableColourRule = {
    key: string;
    value: string;
};

type EditableTrackKindRule = {
    key: string;
    kind: TimelineTrackKind;
};

type DraftSettings = {
    homescreen: {
        useWallTime: boolean;
        colours: EditableColourRule[];
    };
    replay: {
        globalOnlySavedActions: boolean;
        timelineKinds: EditableTrackKindRule[];
        trainerRules: EditableReplayRule[];
    };
};

type SettingsTab = {
    id: "homescreen" | "replay" | "reset";
    label: string;
    badge?: number;
};

const settingsStore = useSettingsStore();
const tracksStore = useTracksStore();
const experimentStore = useExperimentStore();
const modal = ref<HTMLDivElement | null>(null);
let modalInstance: Modal | null = null;
const activeTab = ref<SettingsTab["id"]>("homescreen");
const draft = ref<DraftSettings>(createDraft());
const newTimelineKindRule = ref<EditableTrackKindRule>({ key: "", kind: "numeric" });
const newTrainerReplayRule = ref<EditableReplayRule>({ key: "", value: true });

const tabs = computed<SettingsTab[]>(() => [
    { id: "homescreen", label: "Homescreen", badge: draft.value.homescreen.colours.length || undefined },
    {
        id: "replay",
        label: "Replay",
        badge: draft.value.replay.timelineKinds.length + draft.value.replay.trainerRules.length || undefined,
    },
    { id: "reset", label: "Reset" },
]);

const hasChanges = computed(() => !deepEqual(snapshotDraft(draft.value), snapshotSettings()));

function matchedTracksForRule(ruleKey: string): string[] {
    const normalizedRule = ruleKey.trim();
    if (normalizedRule.length === 0) {
        return [];
    }
    return Array.from(tracksStore.allTrackLabels).filter((trackLabel) => matchesTrackRuleKey(normalizedRule, trackLabel));
}

function matchCount(ruleKey: string): number {
    return matchedTracksForRule(ruleKey).length;
}

function matchTooltip(ruleKey: string): string {
    const matches = matchedTracksForRule(ruleKey);
    if (matches.length === 0) {
        return "No matches among tracks encountered so far";
    }
    return matches.join("\n");
}

const allTrainerNames = computed(() => {
    const uniqueNames = new Set(experimentStore.trainerNames.map((name) => name.trim()).filter((name) => name.length > 0));
    return Array.from(uniqueNames).sort((left, right) => left.localeCompare(right));
});

function matchedTrainersForRule(ruleKey: string): string[] {
    const normalizedRule = ruleKey.trim();
    if (normalizedRule.length === 0) {
        return [];
    }
    return allTrainerNames.value.filter((trainerName) => matchesTrackRuleKey(normalizedRule, trainerName));
}

function trainerMatchCount(ruleKey: string): number {
    return matchedTrainersForRule(ruleKey).length;
}

function trainerMatchTooltip(ruleKey: string): string {
    const matches = matchedTrainersForRule(ruleKey);
    if (matches.length === 0) {
        return "No matches among trainers encountered so far";
    }
    return matches.join("\n");
}

function createDraft(): DraftSettings {
    const current = settingsStore.settings;
    return {
        homescreen: {
            useWallTime: current.visualization.useWallTime,
            colours: entriesFromStringMap(current.visualization.colours),
        },
        replay: {
            globalOnlySavedActions: current.replay.globalOnlySavedActions,
            timelineKinds: entriesFromTrackKindMap(current.visualization.tracks.defaultKinds),
            trainerRules: entriesFromBooleanMap(current.replay.trainerRules),
        },
    };
}

function snapshotSettings(): unknown {
    const settings = settingsStore.settings;
    return {
        homescreen: {
            useWallTime: settings.visualization.useWallTime,
            colours: settings.visualization.colours,
        },
        replay: {
            globalOnlySavedActions: settings.replay.globalOnlySavedActions,
            timelineKinds: settings.visualization.tracks.defaultKinds,
            trainerRules: settings.replay.trainerRules,
        },
    };
}

function snapshotDraft(value: DraftSettings): unknown {
    return {
        homescreen: {
            useWallTime: value.homescreen.useWallTime,
            colours: stringEntriesToMap(value.homescreen.colours),
        },
        replay: {
            globalOnlySavedActions: value.replay.globalOnlySavedActions,
            timelineKinds: trackKindEntriesToMap(value.replay.timelineKinds),
            trainerRules: booleanEntriesToMap(value.replay.trainerRules),
        },
    };
}

function entriesFromBooleanMap(map: Record<string, boolean>): EditableReplayRule[] {
    return Object.entries(map)
        .map(([key, value]) => ({ key, value }))
        .sort((left, right) => left.key.localeCompare(right.key));
}

function booleanEntriesToMap(entries: EditableReplayRule[]): Record<string, boolean> {
    const result: Record<string, boolean> = {};
    for (const entry of entries) {
        const key = entry.key.trim();
        if (key.length === 0) {
            continue;
        }
        result[key] = entry.value;
    }
    return result;
}

function entriesFromStringMap(map: Record<string, string>): EditableColourRule[] {
    return Object.entries(map)
        .map(([key, value]) => ({ key, value }))
        .sort((left, right) => left.key.localeCompare(right.key));
}

function stringEntriesToMap(entries: EditableColourRule[]): Record<string, string> {
    const result: Record<string, string> = {};
    for (const entry of entries) {
        const key = entry.key.trim();
        if (key.length === 0) {
            continue;
        }
        result[key] = entry.value;
    }
    return result;
}

function entriesFromTrackKindMap(map: Record<string, TimelineTrackKind>): EditableTrackKindRule[] {
    return Object.entries(map)
        .map(([key, kind]) => ({ key, kind }))
        .sort((left, right) => left.key.localeCompare(right.key));
}

function trackKindEntriesToMap(entries: EditableTrackKindRule[]): Record<string, TimelineTrackKind> {
    const result: Record<string, TimelineTrackKind> = {};
    for (const entry of entries) {
        const key = entry.key.trim();
        if (key.length === 0) {
            continue;
        }
        result[key] = entry.kind;
    }
    return result;
}

function stableSort(value: unknown): unknown {
    if (Array.isArray(value)) {
        return value.map((entry) => stableSort(entry));
    }
    if (value != null && typeof value === "object") {
        const entries = Object.entries(value as Record<string, unknown>).sort(([left], [right]) => left.localeCompare(right));
        return Object.fromEntries(entries.map(([key, entry]) => [key, stableSort(entry)]));
    }
    return value;
}

function deepEqual(left: unknown, right: unknown): boolean {
    return JSON.stringify(stableSort(left)) === JSON.stringify(stableSort(right));
}

function showModal() {
    draft.value = createDraft();
    resetPendingRows();
    activeTab.value = "homescreen";
    if (modalInstance == null && modal.value != null) {
        modalInstance = new Modal(modal.value);
    }
    modalInstance?.show();
}

function close() {
    if (hasChanges.value && !confirm("Discard unsaved settings changes?")) {
        return;
    }
    modalInstance?.hide();
    draft.value = createDraft();
    resetPendingRows();
}

function save() {
    settingsStore.setReplaySettings({
        globalOnlySavedActions: draft.value.replay.globalOnlySavedActions,
        trainerRules: booleanEntriesToMap(draft.value.replay.trainerRules),
    });

    settingsStore.setVisualizationSettings({
        colours: stringEntriesToMap(draft.value.homescreen.colours),
        tracks: {
            defaultKinds: trackKindEntriesToMap(draft.value.replay.timelineKinds),
        },
        useWallTime: draft.value.homescreen.useWallTime,
    });

    modalInstance?.hide();
}

function resetAll() {
    if (!confirm("Reset all settings to defaults?")) {
        return;
    }
    settingsStore.resetSettings();
    draft.value = createDraft();
    resetPendingRows();
}

function resetPendingRows() {
    newTimelineKindRule.value = { key: "", kind: "numeric" };
    newTrainerReplayRule.value = { key: "", value: true };
}

function addColourRule() {
    draft.value.homescreen.colours.push({ key: "", value: "#808080" });
}

function removeColourRule(index: number) {
    draft.value.homescreen.colours.splice(index, 1);
}

function clearColours() {
    draft.value.homescreen.colours = [];
}

function commitNewTimelineKindRule() {
    const key = newTimelineKindRule.value.key.trim();
    if (key.length === 0) {
        return;
    }

    const existingIndex = draft.value.replay.timelineKinds.findIndex((rule) => rule.key.trim() === key);
    if (existingIndex >= 0) {
        draft.value.replay.timelineKinds[existingIndex] = { key, kind: newTimelineKindRule.value.kind };
    } else {
        draft.value.replay.timelineKinds.push({ key, kind: newTimelineKindRule.value.kind });
    }

    newTimelineKindRule.value = { key: "", kind: newTimelineKindRule.value.kind };
}

function removeTrackKindRule(index: number) {
    draft.value.replay.timelineKinds.splice(index, 1);
}

function commitNewTrainerReplayRule() {
    const key = newTrainerReplayRule.value.key.trim();
    if (key.length === 0) {
        return;
    }

    const existingIndex = draft.value.replay.trainerRules.findIndex((rule) => rule.key.trim() === key);
    if (existingIndex >= 0) {
        draft.value.replay.trainerRules[existingIndex] = { key, value: newTrainerReplayRule.value.value };
    } else {
        draft.value.replay.trainerRules.push({ key, value: newTrainerReplayRule.value.value });
    }

    newTrainerReplayRule.value = { key: "", value: newTrainerReplayRule.value.value };
}

function removeReplayTrainerRule(index: number) {
    draft.value.replay.trainerRules.splice(index, 1);
}

defineExpose({ showModal });
</script>

<style scoped>
.settings-modal {
    border: 1px solid var(--bs-border-color);
    border-radius: 0.9rem;
}

.settings-layout {
    display: grid;
    grid-template-columns: 17rem minmax(0, 1fr);
    gap: 1rem;
}

.settings-nav {
    display: grid;
    align-content: start;
    gap: 0.4rem;
    padding: 0.3rem;
    border: 1px solid var(--bs-border-color);
    border-radius: 0.75rem;
    background: var(--bs-secondary-bg);
}

.settings-tab {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    width: 100%;
    border: 0;
    border-radius: 0.55rem;
    background: transparent;
    color: var(--bs-body-color);
    text-align: left;
    padding: 0.75rem 0.8rem;
}

.settings-tab.active {
    background: var(--bs-body-bg);
    box-shadow:
        0 1px 0 rgba(0, 0, 0, 0.03),
        0 8px 18px rgba(15, 23, 42, 0.06);
    font-weight: 700;
}

.settings-tab-title {
    min-width: 0;
}

.settings-tab-badge {
    flex: 0 0 auto;
}

.settings-content {
    min-width: 0;
}

.settings-modal-subtitle {
    font-size: 0.88rem;
}

.settings-section {
    padding: 0.9rem 0;
    border-bottom: 1px solid var(--bs-border-color);
    display: grid;
    gap: 0.75rem;
}

.settings-section:last-child {
    border-bottom: 0;
    padding-bottom: 0;
}

.settings-section--danger {
    background: color-mix(in srgb, var(--bs-danger-bg-subtle, #f8d7da) 18%, transparent);
    padding: 0.75rem;
    border-radius: 0.6rem;
}

.settings-panel {
    min-height: 100%;
}

.settings-subsection {
    display: grid;
    gap: 0.65rem;
}

.settings-subsection-title {
    font-weight: 700;
}

.settings-divider {
    margin: 0.15rem 0;
    border: 0;
    border-top: 1px solid var(--bs-border-color);
    opacity: 0.9;
}

.section-heading {
    display: flex;
    align-items: start;
    justify-content: space-between;
    gap: 0.75rem;
}

.empty-state {
    border: 1px dashed var(--bs-border-color);
    border-radius: 0.5rem;
    padding: 0.7rem 0.8rem;
    color: var(--bs-secondary-color);
    font-size: 0.9rem;
}

.rule-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 12rem auto;
    gap: 0.5rem;
    align-items: center;
}

.rule-key-cell {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 0.4rem;
    align-items: center;
    min-width: 0;
}

.rule-match-badge {
    cursor: help;
    min-width: 2.1rem;
    text-align: center;
}

.rule-row--color {
    grid-template-columns: 1.2rem minmax(0, 1fr) 3rem minmax(9rem, 14rem) auto;
}

.color-preview {
    width: 1.2rem;
    height: 1.2rem;
    border-radius: 999px;
    border: 1px solid var(--bs-border-color);
}

.color-input {
    padding: 0;
    width: 3rem;
    height: 2.3rem;
}

.color-hex {
    min-width: 0;
}

.settings-switch {
    display: flex;
    gap: 0.6rem;
    align-items: center;
    margin: 0;
}

.settings-reset-actions {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
}

.settings-footer-note {
    font-size: 0.88rem;
}

.rule-key,
.rule-value {
    min-width: 0;
}

@media (max-width: 992px) {
    .settings-layout {
        grid-template-columns: 1fr;
    }

    .settings-nav {
        grid-auto-flow: column;
        grid-auto-columns: minmax(10rem, 1fr);
        overflow-x: auto;
    }

    .rule-row,
    .rule-row--color {
        grid-template-columns: 1fr;
    }
}
</style>
