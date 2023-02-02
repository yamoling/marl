<template>
    <NavBar @navChange="navChange"></NavBar>
    <main>
      <Legacy v-if="selectedMode == 'Training'"></Legacy>
      <TrainingReplay v-else-if="selectedMode == 'Replays'"></TrainingReplay>
    </main>
</template>


<script setup lang="ts">
import { ref } from 'vue';
import Legacy from './components/Legacy.vue';
import NavBar from "./components/NavBar.vue";
import TrainingReplay from './components/TrainingReplay.vue';

const selectedMode = ref("legacy");

function navChange(newActiveTab: string) {
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