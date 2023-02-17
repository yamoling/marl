<template>
  <NavBar ref="nav" @navChange="onNavChange"></NavBar>
  <main>
    <MainTraining v-if="selectedMode == 'Training'"></MainTraining>
    <MainReplay v-else-if="selectedMode == 'Replays'"></MainReplay>
</main>
</template>


<script setup lang="ts">
import { ref } from 'vue';
import NavBar from "./components/NavBar.vue";
import MainTraining from './components/training/MainTraining.vue';
import MainReplay from './components/replay/MainReplay.vue';

interface Nav {
  change: (tabName: NavItems) => void
}

type NavItems = "Training" | "Replays";
const selectedMode = ref("Training" as NavItems);
const nav = ref(null as Nav | null);

function onNavChange(newActiveTab: NavItems) {
  selectedMode.value = newActiveTab;
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
}
</style>