<template>
    <div class="row">
        <div class="col-8">
            <div class="row text-center">
                <AgentInfo v-for="agent in nAgents" :agent-num="agent - 1"
                    :available-actions="episode.available_actions[currentStep][agent - 1]"
                    :qvalues="episode.qvalues[currentStep][agent - 1]" :extras="episode.extras[currentStep][agent - 1]"
                    :obs="episode.obs[currentStep][agent - 1]">
                </AgentInfo>
            </div>
            <div class="row">
                <div class="mx-auto col-auto">
                    <div class="input-group">
                        <button type="button" class="btn btn-success" @click="() => step(-1)"> &lt; </button>
                        <span class="input-group-text"> Current step </span>
                        <input type="text" class="form-control" :value="currentStep" size="5" @keyup.enter="changeStep" />
                        <span class="input-group-text"> / {{ episodeLength }} </span>
                        <button type="button" class="btn btn-success" @click="() => step(1)"> &gt; </button>
                    </div>
                </div>
            </div>
        </div>
        <Rendering class="col-4" :current-image="currentImage" :previous-image="previousImage" :reward="reward">
        </Rendering>
    </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { Episode } from '../../models/Episode';
import { useEpisodeStore } from '../../stores/EpisodeStore';
import AgentInfo from '../AgentInfo.vue';
import Rendering from '../Rendering.vue';


const store = useEpisodeStore();
const episode = ref({} as Episode);
const currentStep = ref(0);
const frames = ref([] as string[]);
const reward = ref(0);


const nAgents = computed(() => (episode.value.qvalues == null) ? 0 : episode.value.qvalues[0].length);
const episodeLength = computed(() => (episode.value.qvalues == null) ? 0 : episode.value.qvalues.length);
const previousImage = computed(() => {
    if (currentStep.value <= 0) {
        return "";
    }
    if (frames.value.length > 0) {
        return frames.value[currentStep.value - 1];
    }
    return "";
});
const currentImage = computed(() => {
    if (frames.value.length > 0) {
        return frames.value[currentStep.value];
    }
    return "";
});


function setEpisode(step: number, num: number) {
    console.debug("Setting episode with step", step, "and number", num);
    currentStep.value = 0;
    store.getTestEpisode(step, num)
        .then(e => {
            console.log(e);
            episode.value = e;
        });
    store.getTestFrames(step, num)
        .then(newFrames => frames.value = newFrames);
}


function step(amount: number) {
    currentStep.value += amount;
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

defineExpose({ setEpisode })
</script>
