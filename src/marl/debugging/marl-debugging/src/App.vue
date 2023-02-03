<template>
    <NavBar ref="nav" @navChange="onNavChange"></NavBar>
    <main>
      <FileExplorer v-if="selectedMode== 'File selection'" path="logs" :isDirectory="true" @fileSelected="fileSelected">
      </FileExplorer>
      <Legacy v-else-if="selectedMode == 'Training'"></Legacy>
      <TrainingReplay v-else-if="selectedMode == 'Replays'"></TrainingReplay>
    </main>
</template>


<script setup lang="ts">
import { ref } from 'vue';
import FileExplorer from './components/FileExplorer.vue';
import Legacy from './components/Legacy.vue';
import NavBar from "./components/NavBar.vue";
import TrainingReplay from './components/TrainingReplay.vue';
import { HTTP_URL } from './constants';

interface Nav {
  change: (tabName: NavItems) => void
}

type NavItems = "File selection" | "Training" | "Replays";
const selectedMode = ref("File selection" as NavItems);
const nav = ref(null as Nav | null);

function onNavChange(newActiveTab: NavItems) {
  selectedMode.value = newActiveTab;
}

function fileSelected(path: string) {
  fetch(`${HTTP_URL}/load/${path}`)
    .then(() => nav.value?.change("Replays"));
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