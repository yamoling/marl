<template>
  <header class="row mb-1">
    <h1 class="col"> RL Debugger</h1>
    <div class="col input-group">
      <span class="input-group-text">Current experiment</span>
      <datalist id="experiments">
        <option v-for="logdir in replayStore.logdirs" :value="logdir"></option>
      </datalist>
      <input type="text" class="form-control" placeholder="Type to search..." @input="onInput" list="experiments"
        :class="showError ? 'is-invalid' : ''">
      <button class="btn btn-success" type="button" @click="() => createExperiment()">
        <font-awesome-icon icon="fa-solid fa-plus" />
      </button>
    </div>
  </header>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useGlobalState } from '../stores/GlobalState';
import { useReplayStore } from '../stores/ReplayStore';


const replayStore = useReplayStore();
const globalState = useGlobalState();
const showError = ref(false);


function onInput(event: Event) {
  const value = (event.target as HTMLInputElement).value;
  const exists = replayStore.logdirs.includes(value);
  if (!exists) return;
  globalState.logdir = value;
  emits("experimentSelected");
  // Detect when the value has been replaced by an item from the datalist
  // const inputEvent = event as InputEvent;
  // const datalistSelected = inputEvent.inputType == "insertReplacementText" || inputEvent.inputType == null;
}


function createExperiment() {
  globalState.logdir = null;
  emits("createExperiment");
}

const emits = defineEmits(["createExperiment", "experimentSelected"]);

</script>

<style>
.nav-item {
  cursor: pointer;
}
</style>