<template>
  <main>
    <ExperimentConfig id="trainModal" @experiment-created="onExperimentSelected" />
    <Header @create-experiment="createExperiment" />
    <Tabs ref="tabs" @tab-delete="onTabDeleted" />
    <Home v-show="tabs.currentTab == 'Home'" @experiment-selected="onExperimentSelected"
      @experiment-deleted="tabs.deleteTab" @create-experiment="() => modal.show()" />
    <ExperimentMain v-for="logdir in openedLogdirs" v-show="logdir == tabs.currentTab" :logdir="logdir"
      @close-experiment="tabs.deleteTab" />
  </main>
</template>


<script setup lang="ts">
import { onMounted, ref } from 'vue';
import Header from './components/Header.vue';
import Tabs from './components/Tabs.vue';
import type { ITabs } from './components/Tabs.vue';
import Home from './components/Home.vue';
import ExperimentMain from './components/ExperimentMain.vue';
import { useExperimentStore } from './stores/ExperimentStore';
import ExperimentConfig from './components/ExperimentConfig.vue';
import { Modal } from 'bootstrap';


const tabs = ref({} as ITabs);
const openedLogdirs = ref([] as string[]);
const experimentStore = useExperimentStore();
let modal = {} as Modal;

function createExperiment() {
  tabs.value.changeTab("Train");
}

function onExperimentSelected(logdir: string) {
  modal.hide();
  openedLogdirs.value.push(logdir);
  tabs.value.addTab(logdir);
  tabs.value.changeTab(logdir);
}

function onTabDeleted(logdir: string) {
  openedLogdirs.value.splice(openedLogdirs.value.indexOf(logdir), 1);
  experimentStore.unloadExperiment(logdir);
}

onMounted(() => {
  modal = new Modal("#trainModal");
});
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