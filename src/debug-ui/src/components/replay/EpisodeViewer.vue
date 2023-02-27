<template>
        <div class="card">
            <div class="card-header row">
                <h5 class="col-auto"> Episode replay </h5>
                <div class="col text-end">
                    <button type="button" class="btn-close" @click="() => emits('close')"></button>
                </div>
            </div>
            <div class="card-body row">
                <div class="col-8">
                    <div class="row text-center">
                        <AgentInfo v-for="agent in nAgents" :agent-num="agent - 1"
                            :available-actions="episode.available_actions[currentStep]?.[agent - 1]"
                            :qvalues="episode.qvalues[currentStep]?.[agent - 1]"
                            :extras="episode.extras[currentStep]?.[agent - 1]" :obs="episode.obs[currentStep]?.[agent - 1]">
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
                </div>
                <Rendering class="col-4" :current-image="currentImage" :previous-image="previousImage" :reward="reward">
                </Rendering>
            </div>
    </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { ReplayEpisode } from '../../models/Episode';
import AgentInfo from '../visualisation/AgentInfo.vue';
import Rendering from '../visualisation/Rendering.vue';


const currentStep = ref(0);

const reward = computed(() => props.episode?.rewards?.[currentStep.value] || 0);
const props = defineProps<{
    frames: string[]
    episode: ReplayEpisode
}>();
document.addEventListener("keydown", (event) => {
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
})


const nAgents = computed(() => (props?.episode?.qvalues == null) ? 0 : props.episode.qvalues[0].length);
const episodeLength = computed(() => (props?.episode?.qvalues == null) ? 0 : props.episode.qvalues.length);
const previousImage = computed(() => {
    if (currentStep.value <= 0) {
        return "";
    }
    if (props.frames.length > 0) {
        return props.frames[currentStep.value - 1];
    }
    return "";
});
const currentImage = computed(() => {
    if (props.frames.length > 0) {
        return props.frames[currentStep.value];
    }
    return "";
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
        if (isNaN(newValue)) {
            currentStep.value = currentStep.value;
        } else {
            currentStep.value = newValue;
        }
    }
}

function reset() {
    currentStep.value = 0;
}

defineExpose({ reset });
const emits = defineEmits(["close"]);
</script>
