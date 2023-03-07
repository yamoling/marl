<template>
  <main>
    <Header @create-experiment="createExperiment" @experiment-selected="onExperimentSelected" />
    <Tabs ref="tabs" :tabs="tabNames" />
    <MainTraining v-show="tabs.currentTab == 'Train'" />
    <MainReplay v-show="tabs.currentTab == 'Replay'" />
  </main>
</template>


<script setup lang="ts">
import MainTraining from './components/training/MainTraining.vue';
import Header from './components/Header.vue';
import Tabs from './components/Tabs.vue';
import type { ITabs } from './components/Tabs.vue';
import { ref } from 'vue';
import MainReplay from './components/replay/MainReplay.vue';
import { useGlobalState } from './stores/GlobalState';

const tabNames = [
  "Train",
  "Replay"
] as const;

const tabs = ref({} as ITabs<typeof tabNames>);
const globalState = useGlobalState();

function createExperiment() {
  globalState.logdir = null;
  tabs.value.changeTab("Train");
}

function onExperimentSelected() {
  console.log("Experiment selected")
  tabs.value.changeTab("Replay");
}
</script>

<style>
:root {
  font-synthesis: none;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  -webkit-text-size-adjust: 100%;
}


html,
body {
  height: 100%;
  margin: 0;
}

#app {
  height: 100%;
  width: 100%;
  padding: 1%;
  display: flex;
  flex-direction: row;
}

main {
  width: 100%;
  height: 100%;
  padding-left: 0.5%;
  /* overflow: hidden; */
}
</style>