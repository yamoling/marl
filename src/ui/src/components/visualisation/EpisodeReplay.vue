<template>
    <div class="replay-shell">
        <div class="d-flex align-items-center justify-content-between mb-2">
            <h5 class="mb-0">Replay episode </h5>
            <button type="button" class="btn btn-outline-danger btn-sm" @click="() => emits('close')">
                Close
            </button>
        </div>
        <font-awesome-icon v-if="loading" class="mx-auto d-block my-5" icon="spinner" spin
            style="height:100px; width: 100px;" />

        <template v-else-if="episode != null">
            <section class="replay-row top-row">
                <img :src="'data:image/jpg;base64, ' + currentFrame" />
                <aside class="top-right">
                    <ActionPanel :episode="episode" :current-step="currentStep" :action-space="episode.action_space"
                        :n-agents="nAgents" />
                </aside>
            </section>

            <section class="replay-row row-divider">
                <div class="timeline-wrap">
                    <div class="timeline-toolbar">
                        <div class="manual-step-input me-5">
                            Step
                            <input type="text" class="form-control form-control-sm" :value="currentStep" size="4"
                                @keyup.enter="changeStep" />
                            <span class="text-muted">/ {{ episodeLength }}</span>
                        </div>

                        <button type="button" class="btn btn-outline-primary btn-sm" :disabled="episode == null"
                            @click="trackWizardModal?.showModal">
                            Choose tracks
                        </button>
                        <span v-if="tracks.length > 0" class="timeline-toolbar-summary text-secondary">
                            {{ tracks.length }} track{{ tracks.length > 1 ? 's' : '' }} selected
                        </span>
                    </div>

                    <div class="timeline-track-list">
                        <div v-for="(track, index) in tracks" :key="track.label" class="timeline-track-item">
                            <div class="timeline-track-toolbar">
                                <div class="timeline-track-toolbar-left">
                                    <div class="btn-group btn-group-sm timeline-track-order">
                                        <button type="button" class="btn btn-sm btn-outline-danger"
                                            @click.stop="() => tracksStore.remove(props.logdir, track)">
                                            <font-awesome-icon :icon="['fas', 'xmark']" />
                                        </button>
                                        <button type="button"
                                            class="btn btn-outline-secondary timeline-track-order-button"
                                            :disabled="index <= 0"
                                            @click="() => tracksStore.swap(props.logdir, index, index - 1)">
                                            <font-awesome-icon :icon="['fas', 'arrow-up']" />
                                        </button>
                                        <button type="button"
                                            class="btn btn-outline-secondary timeline-track-order-button"
                                            :disabled="index >= tracks.length - 1"
                                            @click="() => tracksStore.swap(props.logdir, index, index + 1)">
                                            <font-awesome-icon :icon="['fas', 'arrow-down']" />
                                        </button>
                                    </div>

                                    <span class="timeline-track-label text-capitalize">
                                        {{ track.label }}
                                    </span>

                                    <select class="form-select form-select-sm timeline-track-kind" :value="track.kind"
                                        :aria-label="`${track.label} representation`"
                                        @change="(event) => onTimelineTrackKindChange(track, event)">
                                        <option value="numeric">Numerical</option>
                                        <option value="categorical">Categorical</option>
                                    </select>
                                </div>
                            </div>
                            <TimelineChartTracks :key="`${track.label}:${track.kind}`" :track="track"
                                :current-step="currentStep" @select-step="selectStep" />
                        </div>
                    </div>
                </div>
            </section>

            <section v-if="experiment != null" class="replay-row row-divider replay-analysis">
                <Accordion>
                    <AccordionPanel value="agents">
                        <AccordionHeader>Agent-wise information</AccordionHeader>
                        <AccordionContent>
                            <div class="agent-details-grid mt-3">
                                <div v-for="agent in nAgents" :key="agent">
                                    <AgentInfo :episode="episode" :agent-num="agent - 1" :current-step="currentStep"
                                        :experiment="experiment" />
                                </div>
                            </div>
                        </AccordionContent>
                    </AccordionPanel>
                </Accordion>
            </section>
            <TrackWizard ref="trackWizardModal" :logdir="logdir" :episode="episode" />
        </template>
    </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, shallowRef } from 'vue';
import AgentInfo from './AgentInfo.vue';
import { ReplayEpisode } from '../../models/Episode';
import { useReplayStore } from '../../stores/ReplayStore';
import { Experiment } from '../../models/Experiment';
import Accordion from 'primevue/accordion';
import AccordionPanel from 'primevue/accordionpanel';
import AccordionHeader from 'primevue/accordionheader';
import AccordionContent from 'primevue/accordioncontent';
import ActionPanel from './action/ActionPanel.vue';
import TimelineChartTracks from './TimelineChartTracks.vue';
import TrackWizard from '../modals/TrackWizard.vue';
import { useTracksStore } from '../../stores/TracksStore';
import { Track, type TimelineTrackKind } from '../../models/Timeline';

const trackWizardModal = ref<InstanceType<typeof TrackWizard> | null>(null);
const props = defineProps<{
    logdir: string
    experiment: Experiment | null
}>();

const replayStore = useReplayStore();
const tracksStore = useTracksStore();
const episode = shallowRef<ReplayEpisode | null>(null);
const tracks = computed(() => {
    const e = episode.value;
    if (e === null) return []

    return tracksStore
        .forLogdir(props.logdir)
        .map((config) => {
            const track = e.getTrack(config.label);
            if (track == null) {
                return undefined;
            }
            track.kind = config.kind;
            return track;
        })
        .filter((track) => track !== undefined);
})

const loading = ref(false);
const currentStep = ref(0);
const nAgents = computed(() => episode.value?.episode.actions[0]?.length ?? 0);
const episodeLength = computed(() => episode.value?.length() || 0);
const maxStep = computed(() => Math.max(0, episodeLength.value));
const safeStep = computed(() => {
    if (episodeLength.value === 0) return 0;
    return Math.max(0, Math.min(episodeLength.value, currentStep.value));
});

const currentFrame = computed(() => episode.value?.frameAt(safeStep.value));

const emits = defineEmits<{
    (e: 'close'): void,
}>();


function isEditableTarget(target: EventTarget | null): boolean {
    if (!(target instanceof HTMLElement)) return false;
    return target.matches('input, textarea, select, [contenteditable="true"]');
}

function onKeyDown(event: KeyboardEvent) {
    if (isEditableTarget(event.target)) return;

    switch (event.key) {
        case 'Home':
            event.preventDefault();
            selectStep(0);
            break;
        case 'End':
            event.preventDefault();
            selectStep(maxStep.value);
            break;
        case 'ArrowLeft':
        case 'ArrowUp':
            event.preventDefault();
            step(-1);
            break;
        case 'ArrowRight':
        case 'ArrowDown':
            event.preventDefault();
            step(1);
            break;
        default:
            return;
    }
}

onMounted(() => window.addEventListener('keydown', onKeyDown));
onUnmounted(() => window.removeEventListener('keydown', onKeyDown));

function step(amount: number) {
    selectStep(currentStep.value + amount);
}

function selectStep(step: number) {
    currentStep.value = Math.max(0, Math.min(maxStep.value, step));
}

function changeStep(event: KeyboardEvent) {
    const target = event.target as HTMLInputElement;
    if (target.value === '') {
        currentStep.value = Math.max(0, Math.min(maxStep.value, currentStep.value));
        return;
    }

    const newValue = parseInt(target.value, 10);
    if (!Number.isNaN(newValue)) {
        currentStep.value = Math.max(0, Math.min(maxStep.value, newValue));
    }
}

function onTimelineTrackKindChange(track: Track, event: Event) {
    const kind = (event.target as HTMLSelectElement).value as TimelineTrackKind;
    track.kind = kind;
    tracksStore.update(props.logdir, { label: track.label, kind });
}

async function load(directory: string) {
    episode.value = null;
    loading.value = true;
    const ep = await replayStore.getEpisode(directory);
    episode.value = ep;
    currentStep.value = 0;
    loading.value = false;
}

defineExpose({
    load,
});


</script>

<style scoped>
.replay-shell {
    display: flex;
    flex-direction: column;
    gap: 0;
}

.replay-row {
    padding: 0.65rem 0;
}

.row-divider {
    border-top: 1px solid var(--bs-border-color);
}

.top-row {
    display: grid;
    grid-template-columns: minmax(0, 1.6fr) minmax(360px, 1.25fr);
    gap: 0.75rem;
    align-items: start;
}

.top-right {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
}

.timeline-wrap {
    min-width: 0;
    user-select: none;
}

.timeline-toolbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.65rem;
}

.timeline-toolbar-summary {
    line-height: 1.2;
}

.track-visibility {
    display: flex;
    flex-wrap: wrap;
    gap: 0.8rem;
}

.track-toggle {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.82rem;
    color: var(--bs-secondary-color);
}

.timeline-track-list {
    display: grid;
    gap: 0.35rem;
}

.timeline-track-item {
    display: grid;
    gap: 0.2rem;
    padding: 0.1rem 0;
}

.timeline-track-toolbar {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 0.35rem;
}

.timeline-track-toolbar-left {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.45rem;
    min-width: 0;
}

.timeline-track-order {
    flex-shrink: 0;
}

.timeline-track-order-button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.75rem;
    padding-inline: 0;
}

.timeline-track-label {
    font-size: 0.78rem;
    color: var(--bs-secondary-color);
}

.timeline-track-kind {
    width: auto;
    min-width: 8.5rem;
}

.timeline-track-value {
    margin-left: 0.35rem;
    color: var(--bs-body-color);
    font-weight: 600;
}

.timeline-track-item :deep(.timeline-chart-canvas-shell) {
    min-height: 88px;
}

.manual-step-input {
    display: flex;
    align-items: center;
    gap: 0.2rem;
}

.manual-step-input input {
    width: 4rem;
}

.agent-details-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.75rem;
}

@media (max-width: 992px) {
    .agent-details-grid {
        grid-template-columns: minmax(0, 1fr);
    }
}

@media (max-width: 1200px) {
    .top-row {
        grid-template-columns: minmax(0, 1fr);
    }
}
</style>
