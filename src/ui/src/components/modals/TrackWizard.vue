<template>
    <div ref="modal" class="modal fade" tabindex="-1">
        <div class="modal-dialog modal-dialog-centered modal-lg modal-dialog-scrollable">
            <div class="modal-content timeline-modal">
                <div class="modal-header">
                    <div>
                        <h5 class="modal-title mb-1">Configure timeline tracks to show</h5>
                    </div>
                    <button type="button" class="btn-close" @click="close" aria-label="Close"></button>
                </div>

                <div class="modal-body timeline-body">
                    <div class="wizard-panel">
                        <div class="wizard-toolbar mb-3">
                            <button type="button" class="btn btn-sm btn-outline-secondary"
                                @click="() => draftSelections = []">
                                Clear all
                            </button>
                        </div>

                        <div class="option-list">
                            <section v-for="track in availableTracks" :key="track.label" class="option-card">
                                <template v-if="isGroup(track)">
                                    <div class=" option-card-head">
                                        <label class="track-toggle">
                                            <input class="form-check-input" type="checkbox" :checked="isSelected(track)"
                                                @change="(event) => onGroupToggle(track, event)" />
                                            <span class="option-label text-capitalize"
                                                :class="{ dimmed: !hasAnySelected(track) }">
                                                {{ track.label }}
                                            </span>
                                        </label>

                                        <select class="form-select form-select-sm" :disabled="!isSelected(track)"
                                            :value="track.getMajorityKind()"
                                            @change="(event) => onFamilyKindChange(track, event)">
                                            <option value="numeric">Numerical</option>
                                            <option value="categorical">Categorical</option>
                                        </select>

                                    </div>
                                    <div class="component-grid">
                                        <label v-for="subtrack in track.subTracks" :key="subtrack.label"
                                            class="component-option text-capitalize">
                                            <input class="form-check-input" type="checkbox"
                                                :checked="isSelected(subtrack)"
                                                @change="(event) => onTrackToggle(subtrack, event)" />
                                            <span>{{ subtrack.label }}</span>
                                        </label>
                                    </div>

                                </template>
                                <template v-else>
                                    <div class="option-card-head">
                                        <label class="track-toggle">
                                            <input class="form-check-input" type="checkbox" :checked="isSelected(track)"
                                                @change="(event) => onTrackToggle(track, event)" />
                                            <span class="option-label">{{ track.label }}</span>
                                        </label>
                                        <select class="form-select form-select-sm" :disabled="!isSelected(track)"
                                            :value="kindForTrack(track)"
                                            @change="(event) => onTrackKindChange(track, event)">
                                            <option value="numeric">Numerical</option>
                                            <option value="categorical">Categorical</option>
                                        </select>
                                    </div>
                                </template>



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
import { type TrackConfig, TrackGroup, type TimelineTrackKind } from '../../models/Timeline';
import { useTracksStore } from '../../stores/TracksStore';

const props = defineProps<{
    readonly episode: ReplayEpisode,
    readonly logdir: string,
}>();


const trackStore = useTracksStore();
const modal = ref({} as HTMLDivElement);
let modalInstance: Modal | null = null;
const draftSelections = ref([] as TrackConfig[]);
const availableTracks = computed(() => props.episode.tracks);


function isSelected(track: TrackConfig | TrackGroup): boolean {
    if (track instanceof TrackGroup) {
        return track.subTracks.every(t => isSelected(t));
    }
    return draftSelections.value.some(t => t.label === track.label);
}

function hasAnySelected(track: TrackGroup): boolean {
    return track.subTracks.some((subTrack) => isSelected(subTrack));
}

function isGroup(track: TrackConfig | TrackGroup): track is TrackGroup {
    return track instanceof TrackGroup;
}

function showModal() {
    draftSelections.value = trackStore.selectedTracks[props.logdir].map((track) => ({ label: track.label, kind: track.kind }));
    if (modalInstance == null) {
        modalInstance = new Modal(modal.value);
    }

    modalInstance.show();
}

function close() {
    modalInstance?.hide();
}


function confirm() {
    trackStore.set(props.logdir, draftSelections.value.map((track) => ({ label: track.label, kind: track.kind })));
    close();
}

function onFamilyKindChange(track: TrackGroup, event: Event) {
    const newKind = (event.target as HTMLSelectElement).value as TimelineTrackKind;
    for (const subTrack of track.getTracks()) {
        setDraftKind(subTrack, newKind);
    }
}

function onGroupToggle(track: TrackGroup, event: Event) {
    const isChecked = (event.target as HTMLInputElement).checked;
    for (const subTrack of track.getTracks()) {
        onTrackToggle(subTrack, event);
    }
}

function onTrackKindChange(track: TrackConfig, event: Event) {
    const newKind = (event.target as HTMLSelectElement).value as TimelineTrackKind;
    setDraftKind(track, newKind);
}

function onTrackToggle(track: TrackConfig, event: Event) {
    const isChecked = (event.target as HTMLInputElement).checked;
    if (isChecked) {
        draftSelections.value.push({ label: track.label, kind: track.kind });
    } else {
        draftSelections.value = draftSelections.value.filter((entry) => entry.label !== track.label);
    }
}

function setDraftKind(track: TrackConfig, kind: TimelineTrackKind) {
    const existing = draftSelections.value.find((entry) => entry.label === track.label);
    if (existing == null) {
        return;
    }
    existing.kind = kind;
}

function kindForTrack(track: TrackConfig): TimelineTrackKind {
    return draftSelections.value.find((entry) => entry.label === track.label)?.kind ?? track.kind;
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
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
}

.track-toggle {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    min-width: 0;
    flex: 1 1 auto;
}

.option-card-head .form-select {
    width: auto;
    min-width: 10rem;
}

.option-label {
    font-weight: 700;
}

.component-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr));
    gap: 0.35rem 0.75rem;
}

.component-option {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    min-width: 0;
}

.option-label.dimmed {
    opacity: 0.55;
}

.empty-state {
    padding: 0.9rem 1rem;
    border: 1px dashed var(--bs-border-color);
    border-radius: 0.7rem;
    color: var(--bs-secondary-color);
    background: color-mix(in srgb, var(--bs-body-bg) 96%, var(--bs-secondary) 4%);
}
</style>