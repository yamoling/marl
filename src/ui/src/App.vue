<template>
  <main>
    <header class="row mb-1">
      <h1 class="col"> Experiment manager</h1>
    </header>
    <Tabs ref="tabs" @tab-delete="onTabDeleted" @tab-change="onTabChanged" />
    <Home v-show="tabs.currentTab == 'Home'" @experiment-selected="onExperimentSelected"
      @experiment-deleted="tabs.deleteTab" @compare-experiments="compareExperiments" />
    <ExperimentMain v-for="logdir in openedLogdirs" v-show="logdir == tabs.currentTab" :logdir="logdir"
      @close-experiment="tabs.deleteTab" />
    <ExperimentComparison v-show="tabs.currentTab == 'Compare'" @load-experiment="onExperimentSelected" />
    <footer class="row">
      <SystemInfo />
    </footer>
  </main>
</template>


<script setup lang="ts">
import { ref } from 'vue';
import Tabs from './components/Tabs.vue';
import type { ITabs } from './components/Tabs.vue';
import Home from './components/Home.vue';
import ExperimentMain from './components/ExperimentMain.vue';
import { useExperimentStore } from './stores/ExperimentStore';
import ExperimentComparison from './components/comparison/Main.vue';
import SystemInfo from './components/SystemInfo.vue';

const tabs = ref({} as ITabs);
const openedLogdirs = ref([] as string[]);
const experimentStore = useExperimentStore();


function onExperimentSelected(logdir: string) {
  openedLogdirs.value.push(logdir);
  tabs.value.addTab(logdir);
  tabs.value.changeTab(logdir);
}

function onTabDeleted(logdir: string) {
  if (logdir == "Compare") return
  openedLogdirs.value.splice(openedLogdirs.value.indexOf(logdir), 1);
  experimentStore.unloadExperiment(logdir);
}

function onTabChanged(tabName: string) {
  if (tabName == "Home") {
    experimentStore.refresh();
  }
}

function compareExperiments() {
  tabs.value.changeTab("Compare");
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

footer {
  position: fixed;
  bottom: 0;
  font-size: smaller;
  width: 100%;
}

dialog {
  top: 50%;
  left: 50%;
  translate: -50% -50%;
  border-radius: 3%;
}
</style>