<template>
  <main class="container-fluid">
    <h1>Rl Debugger</h1>
    <div class="grid">
      <AgentInfo v-for="agent in nAgents" ref="agents" :agent-num="agent - 1"></AgentInfo>
      <Rendering :current-image="currentImage" :previous-image="previousImage" :reward="reward"></Rendering>
    </div>
    <div class="container" style="padding-top: 0.5%;">
      <div class="grid">
        <button href="#" role="button" @click="step" :aria-busy="freezeButtons"> Step </button>
        <button href="#" role="button" @click="fastForward" :aria-busy="freezeButtons">Fast forward</button>
        <button href="#" role="button" @click="skip" :aria-busy="freezeButtons">Skip</button>
        <input size="5" type="number" v-model="forwardAmount" />
      </div>
    </div>

  </main>
</template>


<script setup lang="ts">
import { ref } from 'vue';
import AgentInfo from './components/AgentInfo.vue';
import Rendering from './components/Rendering.vue';

interface Agent {
  update: (observation: number[], extras: number[], qvalues: number[], availableActions: number[]) => void
}

const ws = new WebSocket("ws://localhost:5172")
const agents = ref<Agent[] | null>(null);
const nAgents = ref(0);
const forwardAmount = ref(1);
const freezeButtons = ref(false);
const previousImage = ref("");
const currentImage = ref("");
const reward = ref(0);
let done = false;

ws.onopen = function (event: Event) {
  console.log("Connected: ", event);
}

interface ServerUpdate {
  qvalues: number[][],
  observations: number[][],
  state: number[],
  extras: number[][],
  done: boolean
  reward: number,
  available: number[][],
  b64_rendering: any,
}
ws.onmessage = function (event: MessageEvent) {
  const data: ServerUpdate = JSON.parse(event.data);
  nAgents.value = data.qvalues.length;
  previousImage.value = JSON.parse(JSON.stringify(currentImage.value));
  currentImage.value = data.b64_rendering;
  freezeButtons.value = false;
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
    freezeButtons.value = true;
    ws.send(JSON.stringify({ command: "step" }));
  }
}

function fastForward() {
  freezeButtons.value = true;
  ws.send(JSON.stringify({ command: "fastForward", amount: forwardAmount.value - 1 }));
  step()
}

function skip() {
  freezeButtons.value = true;
  ws.send(JSON.stringify({ command: "skip", amount: forwardAmount.value - 1 }));
  step()
}
</script>
