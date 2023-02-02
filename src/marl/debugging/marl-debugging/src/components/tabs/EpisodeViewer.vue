<template>
    <div class="row">
        <div class="col-8">
            <AgentInfo v-for="agent in nAgents" ref="agents" :agent-num="agent - 1"></AgentInfo>
            <div class="row">
                <div class="col-1">
                    <button type="button" class="btn btn-success" @click="step"> Step </button>
                </div>
                <div class="col-3">
                    <div class="input-group mb-3">
                        <button class="btn btn-outline-danger" href="#" @click="skip" type="button"
                            id="button-addon1">Skip</button>
                        <input class="form-control" type="text" v-model="forwardAmount">
                    </div>
                </div>
            </div>
        </div>
        <Rendering class="col-4" :current-image="currentImage" :previous-image="previousImage" :reward="reward">
        </Rendering>
    </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { Episode } from '../../models/Episode';
import { useEpisodeStore } from '../../stores/EpisodeStore';
import type { ServerUpdate } from '../../models/ServerUpdate';
import AgentInfo from '../AgentInfo.vue';
import Rendering from '../Rendering.vue';

interface Props {
    kind: "train" | "test",
    episodeNum: number
}


interface Agent {
    update: (observation: number[], extras: number[], qvalues: number[], availableActions: number[]) => void
}


const store = useEpisodeStore();
const props = defineProps<Props>();
const episode = ref({} as Episode);
const ws = new WebSocket("ws://localhost:5172")
const agents = ref<Agent[] | null>(null);
const nAgents = ref(0);
const forwardAmount = ref(1);
const disableButtons = ref(false);
const previousImage = ref("");
const currentImage = ref("");
const images = ref([] as string[]);
const reward = ref(0);
let done = false;


onMounted(async () => {
    episode.value = await store.getEpisode(props.kind, props.episodeNum);
    const resp = await fetch("http://0.0.0.0:5171/frames/" + props.kind + "/" + props.episodeNum);
    images.value = await resp.json();
});


ws.onopen = function (event: Event) {
    console.log("Connected: ", event);
}


ws.onmessage = function (event: MessageEvent) {
    const data: ServerUpdate = JSON.parse(event.data);
    nAgents.value = data.qvalues.length;
    previousImage.value = JSON.parse(JSON.stringify(currentImage.value));
    currentImage.value = data.b64_rendering;
    disableButtons.value = false;
    reward.value = data.reward;
    done = data.done;
    // Timeout necessary because otherwise 'agents.value' has not yet changed
    setTimeout(() => agents.value?.forEach((agent, num) => {
        agent.update(data.observations[num], data.extras[num], data.qvalues[num], data.available[num]);
    }), 1);
};

ws.onclose = function (e) {
    console.log("Socket is closed.", e.reason);
};


function step() {
    if (done) {
        previousImage.value = currentImage.value;
        currentImage.value = "";
        done = false;
    } else {
        disableButtons.value = true;
        ws.send(JSON.stringify({ command: "step" }));
    }
}

function fastForward() {
    disableButtons.value = true;
    ws.send(JSON.stringify({ command: "fastForward", amount: forwardAmount.value - 1 }));
    step()
}

function skip() {
    disableButtons.value = true;
    ws.send(JSON.stringify({ command: "skip", amount: forwardAmount.value - 1 }));
    step()
}

</script>
