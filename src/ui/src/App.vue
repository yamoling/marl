<template>
  <main>
    <ExperimentConfig id="trainModal" @experiment-created="onExperimentSelected" />
    <header class="row mb-1">
      <h1 class="col"> Experiment manager</h1>
    </header>
    <Tabs ref="tabs" @tab-delete="onTabDeleted" @tab-change="onTabChanged" />
    <Home v-show="tabs.currentTab == 'Home'" @experiment-selected="onExperimentSelected"
      @experiment-deleted="tabs.deleteTab" @create-experiment="() => modal.show()"
      @compare-experiments="compareExperiments" />
    <ExperimentMain v-for="logdir in openedLogdirs" v-show="logdir == tabs.currentTab" :logdir="logdir"
      @close-experiment="tabs.deleteTab" />
    <ExperimentComparison ref="comparison" v-show="tabs.currentTab == 'Compare'" />
  </main>
</template>


<script setup lang="ts">
import { onMounted, ref } from 'vue';
import Tabs from './components/Tabs.vue';
import type { ITabs } from './components/Tabs.vue';
import Home from './components/Home.vue';
import ExperimentMain from './components/ExperimentMain.vue';
import { useExperimentStore } from './stores/ExperimentStore';
import ExperimentConfig from './components/modals/ExperimentConfig.vue';
import { Modal } from 'bootstrap';
import ExperimentComparison from './components/charts/ExperimentComparison.vue';
import type { IExperimentComparison } from './components/charts/ExperimentComparison.vue';


const tabs = ref({} as ITabs);
const comparison = ref({} as IExperimentComparison);
const openedLogdirs = ref([] as string[]);
const experimentStore = useExperimentStore();
let modal = {} as Modal;


function onExperimentSelected(logdir: string) {
  modal.hide();
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
  comparison.value.update([]);
  tabs.value.changeTab("Compare");
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