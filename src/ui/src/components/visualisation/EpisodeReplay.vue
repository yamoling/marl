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
                    <h6 class="mb-2">Actions taken</h6>
                    <div class="meta-line mb-2">
                        <span v-if="currentReward != null" class="badge text-bg-success">
                            Reward: {{ formatNumber(currentReward) }}
                        </span>
                    </div>
                    <div class="agent-action-strip">
                        <span v-for="summary in agentActionSummaries" :key="summary.agentNum"
                            class="badge rounded-pill action-pill">
                            Agent {{ summary.agentNum }}: {{ summary.actionLabel }}
                        </span>
                    </div>
                </aside>
            </section>

            <section class="replay-row row-divider controls-row">
                <button type="button" class="btn btn-success btn-sm" @click="() => step(-1)">
                    <font-awesome-icon icon="fa-solid fa-backward-step" />
                </button>

                <div class="slider-wrap">
                    <input v-model.number="currentStep" type="range" class="form-range mb-1" min="0" :max="maxStep" />
                    <div class="slider-label">Step {{ currentStep }} / {{ episodeLength }}</div>
                </div>

                <button type="button" class="btn btn-success btn-sm" @click="() => step(1)">
                    <font-awesome-icon icon="fa-solid fa-forward-step" />
                </button>

                <div class="manual-step-input">
                    <input type="text" class="form-control form-control-sm" :value="currentStep" size="4"
                        @keyup.enter="changeStep" />
                    <span class="text-muted">/ {{ episodeLength }}</span>
                </div>
            </section>

            <section class="replay-row row-divider replay-analysis">
                <div class="row g-2">
                    <div v-for="agent in nAgents" :key="agent" class="col-12 col-xl-6">
                        <AgentInfo :episode="episode" :agent-num="agent - 1" :current-step="currentStep"
                            :rainbow="rainbow" :experiment="experiment" />
                    </div>
                </div>
            </section>
        </template>
    </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import Rainbow from 'rainbowvis.js';
import AgentInfo from './AgentInfo.vue';
import { ReplayEpisode } from '../../models/Episode';
import { useReplayStore } from '../../stores/ReplayStore';
import { Experiment } from '../../models/Experiment';

const props = defineProps<{
    experiment: Experiment,
    episodeDirectory: string
}>();

const replayStore = useReplayStore();
const loading = ref(false);
const episode = ref(null as ReplayEpisode | null);
const currentStep = ref(0);
const rainbow = new Rainbow();
rainbow.setSpectrum('red', 'yellow', 'olivedrab');

const nAgents = computed(() => (episode.value?.episode.actions[0].length) || 0);
const episodeLength = computed(() => episode.value?.metrics.episode_len || 0);
const maxStep = computed(() => Math.max(0, episodeLength.value));
const safeStep = computed(() => {
    if (episodeLength.value === 0) return 0;
    return Math.max(0, Math.min(episodeLength.value, currentStep.value));
});
const currentFrame = computed(() => episode.value?.frames?.at(safeStep.value) || '');
const currentReward = computed(() => {
    if (episode.value == null || currentStep.value <= 0) return null;
    return episode.value.episode.rewards[currentStep.value - 1] ?? null;
});
const agentActionSummaries = computed(() => {
    if (episode.value == null) return [] as Array<{ agentNum: number, actionLabel: string }>;

    const actionLabels = props.experiment.env.action_space.labels ?? [];
    const actions = episode.value.episode.actions[safeStep.value] ?? [];
    return actions.map((action, index) => ({
        agentNum: index + 1,
        actionLabel: actionLabels[action] ?? `#${action}`,
    }));
});

function isEditableTarget(target: EventTarget | null): boolean {
    if (!(target instanceof HTMLElement)) return false;
    return target.matches('input, textarea, select, [contenteditable="true"]');
}

function onKeyDown(event: KeyboardEvent) {
    if (isEditableTarget(event.target)) return;

    switch (event.key) {
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
    currentStep.value = Math.max(0, Math.min(maxStep.value, currentStep.value + amount));
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

function formatNumber(value: number): string {
    if (value == Math.floor(value)) {
        return value.toString();
    }
    return value.toFixed(3);
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
    grid-template-columns: minmax(0, 2fr) minmax(260px, 1fr);
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
    gap: 0.4rem;
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
    display: grid;
    grid-template-columns: auto minmax(180px, 1fr) auto auto;
    gap: 0.6rem;
    align-items: center;
}

.slider-wrap {
    min-width: 0;
}

.slider-label {
    font-size: 0.8rem;
    color: var(--bs-secondary-color);
}

.manual-step-input {
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.manual-step-input input {
    width: 4rem;
}

@media (max-width: 1200px) {
    .top-row {
        grid-template-columns: minmax(0, 1fr);
    }

    .controls-row {
        grid-template-columns: auto minmax(0, 1fr) auto;
        grid-template-areas:
            "prev slider next"
            "manual manual manual";
    }

    .controls-row>button:first-child {
        grid-area: prev;
    }

    .slider-wrap {
        grid-area: slider;
    }

    .controls-row>button:nth-of-type(2) {
        grid-area: next;
    }

    .manual-step-input {
        grid-area: manual;
    }
}
</style>
