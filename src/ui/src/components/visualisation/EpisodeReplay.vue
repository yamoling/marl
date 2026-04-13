<template>
    <div class="replay-shell">
        <font-awesome-icon v-if="loading" class="mx-auto d-block my-5" icon="spinner" spin
            style="height:100px; width: 100px;" />

        <template v-else-if="episode != null">
            <section class="replay-row top-row">
                <div class="top-left text-center">
                    <img class="img-fluid replay-frame" :src="'data:image/jpg;base64, ' + currentFrame" />
                </div>

                <aside class="top-right">
                    <ActionPanel v-if="episode != null" :episode="episode" :current-step="currentStep"
                        :action-space="resolvedActionSpace" :n-agents="nAgents" />
                </aside>
            </section>

            <section class="replay-row row-divider controls-row">
                <div class="timeline-wrap">
                    <div style="display: flex;">
                        <div class="manual-step-input me-5">
                            Step
                            <input type="text" class="form-control form-control-sm" :value="currentStep" size="4"
                                @keyup.enter="changeStep" />
                            <span class="text-muted">/ {{ episodeLength }}</span>
                        </div>

                        <div class="timeline-resolution">
                            <span>Timeline</span>
                            <div class="btn-group btn-group-sm" role="group" aria-label="Timeline resolution">
                                <button type="button" class="btn btn-outline-secondary"
                                    :class="{ active: timelineMode === 'overview' }" @click="timelineMode = 'overview'">
                                    Overview
                                </button>
                                <button type="button" class="btn btn-outline-secondary"
                                    :class="{ active: timelineMode === 'detail' }" @click="timelineMode = 'detail'">
                                    Detail
                                </button>
                            </div>
                            <span class="text-muted timeline-resolution-hint">
                                {{ timelineMode === 'overview' ? `${timelineBins.length} bins` : 'Full resolution' }}
                            </span>
                        </div>

                    </div>

                    <div class="track-visibility mb-2">
                        <label v-for="toggle in trackToggles" :key="toggle.id" class="track-toggle">
                            <input type="checkbox" :checked="toggle.visible"
                                @change="onTrackToggle(toggle.id, $event)" />
                            {{ toggle.label }}
                        </label>
                    </div>

                    <div class="timeline-track-area" @pointerleave="onTimelinePointerLeave">
                        <span class="now-indicator" :style="nowIndicatorStyle" />

                        <div v-for="track in visibleTracks" :key="track.id" class="timeline-track-row">
                            <span class="timeline-track-label">
                                {{ track.label }}
                                <span class="timeline-track-value">{{ currentTrackValueLabel(track) }}</span>
                            </span>

                            <div class="timeline-track-cells">
                                <button v-for="cell in track.cells" :key="cell.key" type="button" class="timeline-cell"
                                    :class="{
                                        selected: isStepInsideCell(cell),
                                        reward: track.kind === 'continuous-bar',
                                        option: track.kind === 'discrete',
                                    }" :style="timelineCellStyle(track.kind, cell)"
                                    :title="timelineCellTitle(track, cell)" @click="selectStep(cell.representativeStep)"
                                    @pointerdown="onCellPointerDown(cell)" @pointerenter="onCellPointerEnter(cell)">
                                    <span v-if="track.kind === 'continuous-bar'" class="reward-fill"
                                        :style="rewardFillStyle(cell.normalized ?? 0)" />
                                    <span class="visually-hidden">{{ timelineCellTitle(track, cell) }}</span>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            <section class="replay-row row-divider replay-analysis">
                <Accordion class="agent-details-accordion" :value="null">
                    <AccordionPanel value="agents">
                        <AccordionHeader>Agent-wise information</AccordionHeader>
                        <AccordionContent>
                            <div class="agent-details-grid mt-3">
                                <div v-for="agent in nAgents" :key="agent" class="agent-details-item">
                                    <AgentInfo :episode="episode" :agent-num="agent - 1" :current-step="currentStep"
                                        :experiment="experiment" />
                                </div>
                            </div>
                        </AccordionContent>
                    </AccordionPanel>
                </Accordion>
            </section>
        </template>
    </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import Rainbow from 'rainbowvis.js';
import AgentInfo from './AgentInfo.vue';
import { ActionValue, ReplayEpisode } from '../../models/Episode';
import { useReplayStore } from '../../stores/ReplayStore';
import { Experiment } from '../../models/Experiment';
import Accordion from 'primevue/accordion';
import AccordionPanel from 'primevue/accordionpanel';
import AccordionHeader from 'primevue/accordionheader';
import AccordionContent from 'primevue/accordioncontent';
import ActionPanel from './action/ActionPanel.vue';
import { ActionSpace } from '../../models/Env';
import {
    ContinuousBarTrack,
    DiscreteTrack,
    TimelineBin,
    TimelineModel,
    TimelineTrackKind,
} from '../../models/Timeline';

const props = defineProps<{
    experiment: Experiment,
    episodeDirectory: string
}>();

const replayStore = useReplayStore();
const loading = ref(false);
const episode = ref(null as ReplayEpisode | null);
const currentStep = ref(0);
const isScrubbing = ref(false);
const trackVisibility = ref({} as Record<string, boolean>);
const timelineMode = ref<'overview' | 'detail'>('overview');
const rainbow = new Rainbow();
rainbow.setSpectrum('red', 'yellow', 'olivedrab');

type RenderTimelineCell = TimelineBin & {
    value?: number | string;
    normalized?: number;
    category?: string | null;
    colour?: string | null;
};

type RenderTrack = {
    id: string;
    label: string;
    kind: TimelineTrackKind;
    cells: RenderTimelineCell[];
};

const nAgents = computed(() => episode.value?.episode.actions[0]?.length ?? 0);
const episodeLength = computed(() => episode.value?.metrics.episode_len || 0);
const maxStep = computed(() => Math.max(0, episodeLength.value));
const rewardValues = computed(() => episode.value?.episode.rewards ?? []);
const timelineTargetBins = computed(() => {
    if (timelineMode.value === 'detail') return rewardValues.value.length;
    return Math.min(180, Math.max(1, rewardValues.value.length));
});
const safeStep = computed(() => {
    if (episodeLength.value === 0) return 0;
    return Math.max(0, Math.min(episodeLength.value, currentStep.value));
});
const timelineModel = computed(() => new TimelineModel(episodeLength.value));
const timelineBins = computed(() => timelineModel.value.buildBins(rewardValues.value.length, timelineTargetBins.value));
const trackToggles = computed(() => allTracks.value.map((track) => ({
    id: track.id,
    label: track.label,
    visible: trackVisibility.value[track.id] ?? true,
})));
const allTracks = computed(() => {
    const tracks = [] as Array<ContinuousBarTrack | DiscreteTrack>;

    tracks.push(new ContinuousBarTrack('reward', 'Reward', rewardValues.value));

    if (episode.value != null && nAgents.value > 0) {
        for (let agentNum = 0; agentNum < nAgents.value; agentNum++) {
            const optionValues = rewardValues.value.map((_, stepIndex) => {
                const option = episode.value?.action_details[stepIndex]?.options?.[agentNum];
                return normalizeOption(option);
            });
            const hasAnyOption = optionValues.some((option) => option != null);
            if (!hasAnyOption) continue;
            tracks.push(new DiscreteTrack(`options-agent-${agentNum + 1}`, `Option A${agentNum + 1}`, optionValues));
        }
    }

    return tracks;
});
const visibleTracks = computed(() => {
    return allTracks.value
        .filter((track) => trackVisibility.value[track.id] ?? true)
        .map((track): RenderTrack => {
            if (track.kind === 'continuous-bar') {
                return {
                    id: track.id,
                    label: track.label,
                    kind: track.kind,
                    cells: track.buildCells(timelineBins.value),
                };
            }

            if (track.kind === 'discrete') {
                return {
                    id: track.id,
                    label: track.label,
                    kind: track.kind,
                    cells: track.buildCells(timelineBins.value),
                };
            }

            return {
                id: track.id,
                label: track.label,
                kind: track.kind,
                cells: [],
            };
        });
});
const nowIndicatorStyle = computed(() => {
    const ratio = Math.max(0, Math.min(1, timelineModel.value.nowPercent(currentStep.value) / 100));
    return {
        '--now-ratio': ratio.toString(),
    } as Record<string, string>;
});
const currentFrame = computed(() => episode.value?.frames?.at(safeStep.value) || '');
const currentReward = computed(() => {
    if (episode.value == null || currentStep.value <= 0) return null;
    return episode.value.episode.rewards[currentStep.value - 1] ?? null;
});
const resolvedActionSpace = computed<ActionSpace>(() => {
    const replaySpace = episode.value?.action_space;
    const baseSpace = replaySpace ?? props.experiment.env.action_space;
    if (baseSpace.space_type != null) return baseSpace;

    const hasBounds = Array.isArray((baseSpace as { low?: unknown }).low)
        || Array.isArray((baseSpace as { high?: unknown }).high);
    return {
        ...baseSpace,
        space_type: hasBounds ? 'continuous' : 'discrete',
    } as ActionSpace;
});


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

onMounted(() => {
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('pointerup', stopScrubbing);
});

onUnmounted(() => {
    window.removeEventListener('keydown', onKeyDown);
    window.removeEventListener('pointerup', stopScrubbing);
});

watch(
    () => props.episodeDirectory,
    async (newDirectory) => {
        await loadEpisode(newDirectory);
    },
    { immediate: true }
);

watch(
    allTracks,
    (tracks) => {
        const nextVisibility = {} as Record<string, boolean>;
        for (const track of tracks) {
            nextVisibility[track.id] = trackVisibility.value[track.id] ?? true;
        }
        trackVisibility.value = nextVisibility;
    },
    { immediate: true }
);

function step(amount: number) {
    selectStep(currentStep.value + amount);
}

function selectStep(step: number) {
    currentStep.value = Math.max(0, Math.min(maxStep.value, step));
}

function isStepInsideCell(cell: TimelineBin): boolean {
    return currentStep.value >= cell.startStep && currentStep.value <= cell.endStep;
}

function onTrackToggle(trackId: string, event: Event) {
    const target = event.target as HTMLInputElement;
    trackVisibility.value = {
        ...trackVisibility.value,
        [trackId]: target.checked,
    };
}

function onStartPointerDown() {
    isScrubbing.value = true;
    selectStep(0);
}

function onCellPointerDown(cell: TimelineBin) {
    isScrubbing.value = true;
    selectStep(cell.representativeStep);
}

function onCellPointerEnter(cell: TimelineBin) {
    if (!isScrubbing.value) return;
    selectStep(cell.representativeStep);
}

function onTimelinePointerLeave() {
    if (!isScrubbing.value) return;
    // Keep scrub mode active while the pointer is down; it is reset globally on pointerup.
}

function stopScrubbing() {
    isScrubbing.value = false;
}

function rewardFillStyle(normalizedReward: number): { [key: string]: string } {
    const clipped = Math.max(-1, Math.min(1, normalizedReward));
    const style = {
        height: `${Math.abs(clipped) * 50}%`,
    } as { [key: string]: string };

    if (clipped >= 0) {
        style.bottom = '50%';
    } else {
        style.top = '50%';
    }

    return style;
}

function timelineCellStyle(kind: TimelineTrackKind, cell: RenderTimelineCell): { [key: string]: string } {
    if (kind === 'discrete') {
        if (cell.colour == null) {
            return {
                backgroundColor: 'var(--bs-tertiary-bg)',
            };
        }
        return {
            backgroundColor: cell.colour,
        };
    }

    return {};
}

function timelineCellTitle(track: RenderTrack, cell: RenderTimelineCell): string {
    const range = (cell.startStep === cell.endStep)
        ? `Step ${cell.startStep}`
        : `Steps ${cell.startStep}-${cell.endStep}`;

    if (track.kind === 'continuous-bar') {
        return `${track.label} | ${range} | Value ${formatNumber(cell.value ?? 0)}`;
    }

    if (track.kind === 'discrete') {
        return `${track.label} | ${range} | Value ${cell.category ?? 'none'}`;
    }

    return `${track.label} | ${range}`;
}

function currentTrackValueLabel(track: RenderTrack): string {
    if (currentStep.value <= 0) return '-';

    const currentCell = track.cells.find((cell) => isStepInsideCell(cell));
    if (currentCell == null) return '-';

    if (track.kind === 'continuous-bar') {
        return formatNumber(currentCell.value ?? 0);
    }

    if (track.kind === 'discrete') {
        return currentCell.category ?? 'none';
    }

    return '-';
}

function normalizeOption(value: unknown): string | null {
    if (value == null) return null;
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
        return String(value);
    }
    try {
        return JSON.stringify(value);
    } catch {
        return String(value);
    }
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

async function loadEpisode(episodeDirectory: string) {
    episode.value = null;
    loading.value = true;

    const replay = await replayStore.getEpisode(episodeDirectory);
    episode.value = replay;
    currentStep.value = 0;

    const details = replay.action_details.flatMap((detail) => Object.values(detail)).flat(3);
    if (details.length > 0) {
        const min = Math.min(...details);
        const max = Math.max(...details);
        rainbow.setNumberRange(min, max);
    }

    loading.value = false;
}

function formatNumber(value: number | string): string {
    // Convert to number if it's a string
    const numValue = typeof value === 'string' ? parseFloat(value) : value;
    
    // If conversion failed or value is not a number, return string representation
    if (isNaN(numValue)) {
        return String(value);
    }
    
    if (numValue == Math.floor(numValue)) {
        return numValue.toString();
    }
    return numValue.toFixed(3);
}
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

.replay-frame {
    max-height: 34vh;
    object-fit: contain;
}

.top-right {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
}

.agent-action-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
}

.action-pill {
    background: var(--bs-secondary-bg);
    color: var(--bs-body-color);
    border: 1px solid var(--bs-border-color);
    font-weight: 600;
}

.controls-row {
    --track-label-width: 7rem;
    display: block;
}

.timeline-wrap {
    min-width: 0;
    user-select: none;
}

.start-marker {
    width: 1.9rem;
    padding: 0;
    border: 1px solid var(--bs-border-color);
    background: var(--bs-secondary-bg);
    color: var(--bs-body-color);
    font-weight: 700;
}

.start-marker.selected {
    background: color-mix(in srgb, var(--bs-success) 20%, var(--bs-body-bg));
    border-color: var(--bs-success);
}

.timeline-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.5rem;
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

.timeline-track-area {
    position: relative;
    display: grid;
    gap: 0.25rem;
    padding: 0.35rem;
    border: 1px solid var(--bs-border-color);
    border-radius: 0.375rem;
}

.now-indicator {
    position: absolute;
    top: 0.35rem;
    bottom: 0.35rem;
    width: 2px;
    left: calc(var(--track-label-width) + (100% - var(--track-label-width) - 0.7rem) * var(--now-ratio));
    background: color-mix(in srgb, var(--bs-primary) 60%, #000);
    box-shadow: 0 0 0 1px color-mix(in srgb, var(--bs-body-bg) 70%, transparent);
    z-index: 4;
    pointer-events: none;
}

.timeline-track-row {
    display: grid;
    grid-template-columns: var(--track-label-width) minmax(0, 1fr);
    align-items: center;
    gap: 0.45rem;
}

.timeline-track-label {
    font-size: 0.78rem;
    color: var(--bs-secondary-color);
}

.timeline-track-value {
    margin-left: 0.35rem;
    color: var(--bs-body-color);
    font-weight: 600;
}

.timeline-track-cells {
    position: relative;
    display: flex;
    gap: 1px;
    min-height: 20px;
    border: 1px solid var(--bs-border-color);
    border-radius: 0.25rem;
    overflow: hidden;
}

.timeline-cell {
    flex: 1 1 auto;
    min-width: 2px;
    border: 0;
    background: color-mix(in srgb, var(--bs-body-color) 8%, transparent);
    position: relative;
    padding: 0;
}

.timeline-cell:hover {
    background: color-mix(in srgb, var(--bs-success) 16%, transparent);
}

.timeline-cell.selected {
    outline: 1px solid var(--bs-success);
    outline-offset: -1px;
    background: color-mix(in srgb, var(--bs-success) 24%, transparent);
    z-index: 3;
}

.reward-fill {
    position: absolute;
    left: 10%;
    width: 80%;
    border-radius: 2px;
    background: linear-gradient(to top,
            color-mix(in srgb, var(--bs-danger) 22%, transparent),
            color-mix(in srgb, var(--bs-success) 42%, transparent));
    z-index: 2;
}

.slider-label {
    font-size: 0.8rem;
    color: var(--bs-secondary-color);
}

.manual-step-input {
    display: flex;
    align-items: center;
    gap: 0.2rem;
}

.manual-step-input input {
    width: 4rem;
}

.timeline-resolution {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
}


.timeline-resolution-hint {
    font-size: 0.8rem;
}

.agent-details-accordion {
    width: 100%;
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

.agent-details-item {
    min-width: 0;
}

@media (max-width: 1200px) {
    .top-row {
        grid-template-columns: minmax(0, 1fr);
    }

    .controls-row {
        --track-label-width: 5.25rem;
    }

    .timeline-header {
        flex-direction: column;
        align-items: flex-start;
    }
}
</style>
