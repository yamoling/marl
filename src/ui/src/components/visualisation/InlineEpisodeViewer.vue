<template>
    <div class="inline-replay" tabindex="0" @keydown="onKeyDown">
        <div class="d-flex align-items-center justify-content-between mb-2">
            <h5 class="mb-0">Replay episode {{ episodeDirectory }}</h5>
            <button type="button" class="btn btn-outline-danger btn-sm" @click="emits('close')">Close</button>
        </div>

        <font-awesome-icon v-if="loading" class="mx-auto d-block my-4" icon="spinner" spin
            style="height:64px; width: 64px;" />

        <template v-else-if="episode != null">
            <div class="row mb-2">
                <AgentInfo v-for="agent in nAgents" :key="agent" class="col-6" :episode="episode" :agent-num="agent - 1"
                    :current-step="currentStep" :rainbow="rainbow" :experiment="experiment" />
            </div>

            <div class="row mb-2">
                <div class="mx-auto col-auto">
                    <div class="input-group input-group-sm">
                        <button type="button" class="btn btn-success" @click="() => step(-1)">
                            <font-awesome-icon icon="fa-solid fa-backward-step" />
                        </button>
                        <span class="input-group-text">Step</span>
                        <input type="text" class="form-control" :value="currentStep" size="5"
                            @keyup.enter="changeStep" />
                        <span class="input-group-text"> / {{ episodeLength }} </span>
                        <button type="button" class="btn btn-success" @click="() => step(1)">
                            <font-awesome-icon icon="fa-solid fa-forward-step" />
                        </button>
                    </div>
                </div>
            </div>

            <div class="text-center">
                <p v-if="currentStep > 0">Reward: {{ episode.episode.rewards[currentStep - 1] }}</p>
                <img class="img-fluid frame" :src="'data:image/jpg;base64, ' + currentFrame" />
            </div>
        </template>
    </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import Rainbow from 'rainbowvis.js';
import AgentInfo from './AgentInfo.vue';
import { ReplayEpisode } from '../../models/Episode';
import { useReplayStore } from '../../stores/ReplayStore';
import { Experiment } from '../../models/Experiment';

const props = defineProps<{
    experiment: Experiment,
    episodeDirectory: string
}>();

const emits = defineEmits<{
    (event: 'close'): void
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
const currentFrame = computed(() => episode.value?.frames?.at(currentStep.value) || '');

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

function onKeyDown(event: KeyboardEvent) {
    switch (event.key) {
        case 'ArrowLeft':
        case 'ArrowUp':
            step(-1);
            break;
        case 'ArrowRight':
        case 'ArrowDown':
            step(1);
            break;
        default:
            break;
    }
}

async function loadEpisode(episodeDirectory: string) {
    episode.value = null;
    loading.value = true;

    const replay = await replayStore.getEpisode(episodeDirectory);
    episode.value = replay;
    currentStep.value = 0;

    const details = replay.action_details.flatMap((detail) => Object.values(detail)).flat(3) as number[];
    if (details.length > 0) {
        const min = Math.min(...details);
        const max = Math.max(...details);
        rainbow.setNumberRange(min, max);
    }

    loading.value = false;
}
</script>

<style scoped>
.inline-replay {
    height: 100%;
    overflow: auto;
}

.frame {
    max-height: 48vh;
    object-fit: contain;
}
</style>
