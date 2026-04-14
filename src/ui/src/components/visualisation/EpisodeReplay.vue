<template>
    <div class="replay-shell">
        <font-awesome-icon v-if="loading" class="mx-auto d-block my-5" icon="spinner" spin
            style="height:100px; width: 100px;" />

        <template v-else-if="episode != null">
            <section class="replay-row top-row">
                <img :src="'data:image/jpg;base64, ' + currentFrame" />
                <aside class="top-right">
                    <ActionPanel :episode="episode" :current-step="currentStep" :action-space="resolvedActionSpace"
                        :n-agents="nAgents" />
                </aside>
            </section>

            <section class="replay-row row-divider">
                <div class="timeline-wrap">
                    <div style="display: flex;">
                        <div class="manual-step-input me-5">
                            Step
                            <input type="text" class="form-control form-control-sm" :value="currentStep" size="4"
                                @keyup.enter="changeStep" />
                            <span class="text-muted">/ {{ episodeLength }}</span>
                        </div>

                    </div>

                    <div class="track-visibility mb-2">
                        <label v-for="toggle in trackToggles" :key="toggle.id" class="track-toggle">
                            <input type="checkbox" :checked="toggle.visible"
                                @change="onTrackToggle(toggle.id, $event)" />
                            {{ toggle.label }}
                        </label>
                    </div>

                    <div class="timeline-track-list">
                        <div v-for="(view, index) in visibleTrackViews" :key="view.track.id"
                            class="timeline-track-item">
                            <div class="timeline-track-toolbar">
                                <div class="timeline-track-toolbar-left">
                                    <div class="btn-group btn-group-sm timeline-track-order" role="group"
                                        :aria-label="`${view.track.label} order controls`">
                                        <button type="button"
                                            class="btn btn-outline-secondary timeline-track-order-button"
                                            :disabled="index === 0" @click="moveTrack(view.track.id, -1)">
                                            <font-awesome-icon :icon="['fas', 'arrow-up']" />
                                        </button>
                                        <button type="button"
                                            class="btn btn-outline-secondary timeline-track-order-button"
                                            :disabled="index === visibleTrackViews.length - 1"
                                            @click="moveTrack(view.track.id, 1)">
                                            <font-awesome-icon :icon="['fas', 'arrow-down']" />
                                        </button>
                                    </div>

                                    <span class="timeline-track-label">
                                        {{ view.track.label }}
                                        <span class="timeline-track-value">{{ currentTrackValueLabel(view.track)
                                        }}</span>
                                    </span>
                                </div>
                            </div>
                            <TimelineChartTracks :track="view.track" :current-step="currentStep"
                                @select-step="selectStep" />
                        </div>
                    </div>
                </div>
            </section>

            <section class="replay-row row-divider replay-analysis">
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
        </template>
    </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
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
import { ActionSpace } from '../../models/Env';
import {
    buildReplayTracks,
    currentTrackValueLabelAtStep,
    loadTimelineOrder,
    persistTimelineOrder,
    syncTimelineOrder,
} from './replayTimeline';

const props = defineProps<{
    experiment: Experiment,
    episodeDirectory: string
}>();

const replayStore = useReplayStore();
const loading = ref(false);
const episode = ref(null as ReplayEpisode | null);
const currentStep = ref(0);
const trackVisibility = ref({} as Record<string, boolean>);
const trackOrder = ref<string[]>([]);


const nAgents = computed(() => episode.value?.episode.actions[0]?.length ?? 0);
const episodeLength = computed(() => episode.value?.metrics.episode_len || 0);
const maxStep = computed(() => Math.max(0, episodeLength.value));
const rewardValues = computed(() => episode.value?.episode.rewards ?? []);
const safeStep = computed(() => {
    if (episodeLength.value === 0) return 0;
    return Math.max(0, Math.min(episodeLength.value, currentStep.value));
});
const trackToggles = computed(() => allTracks.value.map((track) => ({
    id: track.id,
    label: track.label,
    visible: trackVisibility.value[track.id] ?? true,
})));
const allTracks = computed(() => buildReplayTracks(episode.value, rewardValues.value, nAgents.value));
const timelineLayoutStorageKey = computed(() => `marl.replay.timeline-layout:${props.experiment.logdir}:${props.episodeDirectory}`);
const orderedTrackViews = computed(() => {
    const trackById = new Map(allTracks.value.map((track) => [track.id, track] as const));
    const orderedIds = trackOrder.value.length > 0
        ? trackOrder.value
        : allTracks.value.map((track) => track.id);

    return orderedIds
        .map((trackId) => trackById.get(trackId))
        .filter((track) => track != null)
        .map((track) => ({ track }));
});
const visibleTrackViews = computed(() => orderedTrackViews.value.filter((view) => trackVisibility.value[view.track.id] ?? true));
const currentFrame = computed(() => episode.value?.frames?.at(safeStep.value) || '');
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

watch(
    allTracks,
    () => {
        const nextVisibility = {} as Record<string, boolean>;
        for (const track of allTracks.value) {
            nextVisibility[track.id] = trackVisibility.value[track.id] ?? true;
        }
        trackVisibility.value = nextVisibility;
        syncTimelineLayout();
    },
    { immediate: true }
);


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
});

onUnmounted(() => {
    window.removeEventListener('keydown', onKeyDown);
});

watch(
    () => props.episodeDirectory,
    async (newDirectory) => {
        await loadEpisode(newDirectory);
    },
    { immediate: true }
);

function step(amount: number) {
    selectStep(currentStep.value + amount);
}

function selectStep(step: number) {
    currentStep.value = Math.max(0, Math.min(maxStep.value, step));
}

function onTrackToggle(trackId: string, event: Event) {
    const target = event.target as HTMLInputElement;
    trackVisibility.value = {
        ...trackVisibility.value,
        [trackId]: target.checked,
    };
}


function moveTrack(trackId: string, direction: -1 | 1) {
    const currentIndex = trackOrder.value.indexOf(trackId);
    if (currentIndex < 0) return;

    const nextIndex = currentIndex + direction;
    if (nextIndex < 0 || nextIndex >= trackOrder.value.length) return;

    const nextOrder = [...trackOrder.value];
    const [moved] = nextOrder.splice(currentIndex, 1);
    nextOrder.splice(nextIndex, 0, moved);
    trackOrder.value = nextOrder;
    persistTimelineOrder(timelineLayoutStorageKey.value, trackOrder.value);
}

function currentTrackValueLabel(track: (typeof allTracks.value)[number]): string {
    return currentTrackValueLabelAtStep(track, currentStep.value);
}

function syncTimelineLayout() {
    const storedOrder = loadTimelineOrder(timelineLayoutStorageKey.value);
    trackOrder.value = syncTimelineOrder(trackOrder.value, allTracks.value, storedOrder);
    persistTimelineOrder(timelineLayoutStorageKey.value, trackOrder.value);
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

    loading.value = false;
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

.top-right {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
}

.timeline-wrap {
    min-width: 0;
    user-select: none;
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
