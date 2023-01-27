<template>
  <main class="container-fluid">
    <h1>Rl Debugger</h1>
    <div class="grid">
      <AgentInfo v-for="agent in nAgents" ref="agents" :agent-num="agent - 1"></AgentInfo>
      <div>
        <h2>Rendering</h2>
        <div class="grid">
          <div>
            <h3>Previous</h3>
            <img :src="'data:image/jpg;base64, ' + previousImage" />
          </div>
          <div>
            <h3>Current</h3>
            <img :src="'data:image/jpg;base64, ' + currentImage" />
          </div>
        </div>
      </div>
    </div>
    <div class="container" style="padding-top: 0.5%;">
      <div class="grid">
        <button href="#" role="button" @click="step" :aria-busy="freezeButtons"> Step </button>
        <button href="#" role="button" @click="skip" :aria-busy="freezeButtons">Skip</button>
        <input size="5" type="number" v-model="skipAmount" />
      </div>
    </div>

  </main>
</template>


<script setup lang="ts">
import { ref } from 'vue';
import AgentInfo from './components/AgentInfo.vue';

interface Agent {
  update: (observations: number[], extras: number[], qvalues: number[], availableActions: boolean[]) => void
}

const ws = new WebSocket("ws://localhost:5172")
const agents = ref<Agent[] | null>(null);
const nAgents = ref(0);
const skipAmount = ref(10);
const freezeButtons = ref(false);
const previousImage = ref("");
const currentImage = ref("");

ws.onopen = function (event: Event) {
  console.log("Connected: ", event);
}

interface ServerUpdate {
  qvalues: number[][],
  observations: number[][],
  available: boolean[][],
  extras: number[][],
  b64_rendering: any
}
ws.onmessage = function (event: MessageEvent) {
  const data: ServerUpdate = JSON.parse(event.data);
  nAgents.value = data.qvalues.length;
  previousImage.value = JSON.parse(JSON.stringify(currentImage.value));
  currentImage.value = data.b64_rendering;
  // Timeout necessary because otherwise 'agents.value' has not yet changed
  setTimeout(() => agents.value?.forEach((agent, num) => {
    agent.update(data.observations[num], data.extras[num], data.qvalues[num], data.available[num]);
    freezeButtons.value = false;
  }), 1);
};

ws.onclose = function (e) {
  console.log("Socket is closed.", e.reason);
};


function step() {
  ws.send(JSON.stringify({ command: "step" }));
  freezeButtons.value = true;
}

function skip() {
  ws.send(JSON.stringify({ command: "skip", amount: skipAmount.value }));
  freezeButtons.value = true;
}
</script>
