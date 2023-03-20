<template>
    <div ref="modal" class="modal fade" tabindex="-1">
        <div class="modal-dialog modal-dialog-centered modal-xl">
            <div class="modal-content">
                <div class="modal-header">
                    <h5> Episode replay </h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body row">
                    <!-- Loading spinner -->
                    <font-awesome-icon v-if="episode == null" class="mx-auto" icon="spinner" spin
                        style="height:100px; width: 100px;" />
                    <div class="row mb-1 mx-auto">
                        <AgentInfo v-for="agent in nAgents" class="col-6" :episode="episode" :agent-num="agent - 1"
                            :current-step="currentStep" :rainbow="rainbow">
                        </AgentInfo>
                    </div>
                    <div class="row">
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
                    <Rendering class="col-auto mx-auto" :current-image="currentFrame" :previous-image="previousFrame"
                        :reward="reward" />
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-outline-danger" data-bs-dismiss="modal">Close</button>
                </div>

            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import { ReplayEpisode } from '../../models/Episode';
import Rainbow from "rainbowvis.js";
import AgentInfo from '../visualisation/AgentInfo.vue';
import Rendering from '../visualisation/Rendering.vue';


const currentStep = ref(0);
const modal = ref({} as HTMLElement);
const rainbow = new Rainbow();
rainbow.setSpectrum("red", "yellow", "olivedrab")

const reward = computed(() => props.episode?.episode?.rewards?.[currentStep.value] || 0);
const nAgents = computed(() => (props?.episode?.qvalues == null) ? 0 : props.episode.qvalues[0].length);
const episodeLength = computed(() => props.episode?.metrics.episode_length || 0);
const currentFrame = computed(() => props.episode?.frames?.at(currentStep.value) || '');
const previousFrame = computed(() => {
    if (currentStep.value <= 0) {
        return ""
    }
    return props.episode?.frames?.at(currentStep.value - 1) || '';
})
const props = defineProps<{
    frames: string[]
    episode: ReplayEpisode | null
}>();


watch(props, (newProps) => {
    // Get the min and the max of the qvalues
    currentStep.value = 0;
    const episode = newProps.episode;
    if (episode == null) {
        return;
    }
    const minQValue = Math.min(...episode?.qvalues.map(qs => Math.min(...qs.map(q => Math.min(...q)))));
    const maxQValue = Math.max(...episode?.qvalues.map(qs => Math.max(...qs.map(q => Math.max(...q)))));
    rainbow.setNumberRange(minQValue, maxQValue);
});


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
</script>
