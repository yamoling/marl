<template>
    <div ref="modal" class="modal fade" tabindex="-1">
        <div class="modal-dialog modal-dialog-centered modal-xl">
            <div class="modal-content">
                <div class="modal-header">
                    <h5> Replay episode {{ episode?.directory }} </h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body row">
                    <!-- Loading spinner -->
                    <font-awesome-icon v-if="episode == null" class="mx-auto" icon="spinner" spin
                        style="height:100px; width: 100px;" />
                    <div class="row mb-1 mx-auto">
                        <AgentInfo v-for="agent in nAgents" class="col-6" :episode="episode" :agent-num="agent - 1"
                            :current-step="currentStep" :rainbow="rainbow" :experiment="experiment">
                        </AgentInfo>
                    </div>
                    <div class="row mb-2">
                        <div class="mx-auto col-auto">
                            <div class="input-group">
                                <button type="button" class="btn btn-success" @click="() => step(-1)">
                                    <font-awesome-icon icon="fa-solid fa-solid fa-backward-step" />
                                </button>
                                <span class="input-group-text"> Current step </span>
                                <input type="text" class="form-control" :value="currentStep" size="5"
                                    @keyup.enter="changeStep" />
                                <span class="input-group-text"> / {{ episodeLength }} </span>
                                <button type="button" class="btn btn-success" @click="() => step(1)">
                                    <font-awesome-icon icon="fa-solid fa-solid fa-forward-step" />
                                </button>
                            </div>
                        </div>
                    </div>
                    <div class="row">
                        <div v-if="episode" class="col-auto mx-auto">
                            <p v-if="currentStep > 0">
                                Reward: {{ episode.episode.rewards[currentStep - 1] }}
                            </p>
                            <img :src="'data:image/jpg;base64, ' + currentFrame" />
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-outline-danger" data-bs-dismiss="modal">Close</button>
                </div>

            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { ReplayEpisode } from '../../models/Episode';
import Rainbow from "rainbowvis.js";
import AgentInfo from './AgentInfo.vue';
import { useReplayStore } from '../../stores/ReplayStore';
import { Modal } from 'bootstrap';
import { Experiment } from '../../models/Experiment';


const replayStore = useReplayStore();
const loading = ref(false);
const episode = ref(null as ReplayEpisode | null);
const currentStep = ref(0);
const modal = ref({} as HTMLElement);
const rainbow = new Rainbow();
rainbow.setSpectrum("red", "yellow", "olivedrab")
defineProps<{
    experiment: Experiment
}>();

const nAgents = computed(() => (episode.value?.episode.actions[0].length) || 0);
const episodeLength = computed(() => episode.value?.metrics.episode_length || 0);
const currentFrame = computed(() => episode.value?.frames?.at(currentStep.value) || '');

onMounted(() => {
    modal.value.addEventListener("keydown", (event) => {
        switch (event.key) {
            case "ArrowLeft":
            case "ArrowUp":
                step(-1)
                break;
            case "ArrowRight":
            case "ArrowDown":
                step(1);
                break
            default: return;
        }
    });
});

function step(amount: number) {
    currentStep.value = Math.max(0, Math.min(episodeLength.value, currentStep.value + amount));
}

function changeStep(event: KeyboardEvent) {
    const target = event.target as HTMLInputElement;
    if (target.value == "") {
        currentStep.value = currentStep.value;
    } else {
        const newValue = parseInt(target.value);
        if (!isNaN(newValue)) {
            currentStep.value = newValue;
        }
    }
}

async function viewEpisode(episodeDirectory: string) {
    episode.value = null;
    loading.value = true;
    (new Modal(modal.value)).show()
    const replay = await replayStore.getEpisode(episodeDirectory);
    episode.value = replay;
    currentStep.value = 0;
    if (episode.value.qvalues != null && episode.value.qvalues.length > 0) {
        const allQvalues = episode.value.qvalues.flat(4);
        const minQValue = Math.min(...allQvalues);
        const maxQValue = Math.max(...allQvalues);
        rainbow.setNumberRange(minQValue, maxQValue);
    }
    loading.value = false;
}

defineExpose({ viewEpisode });
</script>
