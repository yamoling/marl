<template>
    <div>
        <div class="row mb-2">
            <div class="col-auto">
                <div class="input-group mb-1">
                    <label class="input-group-text">Algorithms</label>
                    <select class="form-select" v-model="selectedAlgo">
                        <option v-for="algo in algoStore.algorithms" :value="algo"> {{ algo }}</option>
                    </select>
                </div>
                <div class="input-group mb-1">
                    <label class="input-group-text">Select a level</label>
                    <select class="form-select" v-model="selectedLevel">
                        <option v-for="algo in algoStore.maps" :value="algo">
                            {{ algo }}
                        </option>
                    </select>
                </div>
            </div>
        </div>

        <fieldset class="row mb-2">
            <div class="col-auto">
                <legend>Algorithm wrappers</legend>
                <div class="form-check form-switch">
                    <label class="form-check-label">
                        <input class="form-check-input" type="checkbox" role="switch" name="n-step" @change="algoWrapperChanged" />
                        N-step return
                        <input type="number" size="5" v-model="nStep" />
                    </label>
                </div>
            </div>
        </fieldset>

        <fieldset class="row mb-2">
            <div class="col-auto">
                <legend>Env wrappers</legend>
                <div v-for="wrapper in algoStore.envWrappers" class="form-check form-switch">
                    <label class="form-check-label">
                        <input class="form-check-input" type="checkbox" role="switch" :name="wrapper"
                            @change="envWrapperChanged" />
                        {{ wrapper }}
                        <input v-if="wrapper == 'TimeLimit'" type="number" size="8" v-model="timeLimitValue" />
                    </label>
                </div>
            </div>
        </fieldset>

        <fieldset class="row mb-2">
            <div class="col-auto">
                <legend>Replay Memory</legend>
                <div class="input-group mb-1">
                    <span class="input-group-text"> Size </span>
                    <input type="text" class="form-control" v-model.number="memorySize" size="2" />
                </div>

                <div class="input-group mb-1">
                    <label class="input-group-text"> Priotitized
                        <input type="checkbox" class="form-check-input m-1 mx-3" v-model="priotitizedMemory" />
                    </label>
                </div>
            </div>
        </fieldset>


        <button v-if="!loading" role="button" class="btn btn-primary" @click="send">
            Start
            <font-awesome-icon icon="fa-solid fa-play" />
        </button>
        <button v-else role="button" class="btn btn-primary" disabled>
            <font-awesome-icon icon="fa-solid fa-spinner" spin />
        </button>
    </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { HTTP_URL } from '../../constants';
import { useAlgorithmStore } from '../../stores/AlgoStore';


const algoStore = useAlgorithmStore();
const loading = ref(false);
const selectedAlgo = ref(algoStore.algorithms[0]);
const selectedLevel = ref(algoStore.maps[0]);
const timeLimitValue = ref(20);
const nStep = ref(5);
const memorySize = ref(10000);
const priotitizedMemory = ref(true);
const envWrappers = [] as string[];
const algoWrappers = [] as string[];

const emits = defineEmits(['start']);

function send() {
    if (selectedAlgo.value == null || selectedLevel.value == null) {
        alert("Please select an algorithm and a level");
        return;
    }
    loading.value = true;
    fetch(`${HTTP_URL}/algo/create`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            algo: selectedAlgo.value,
            wrappers: envWrappers,
            timeLimit: timeLimitValue.value,
            level: selectedLevel.value,
            memory: {
                size: memorySize.value,
                prioritized: priotitizedMemory.value
            }
        })
    })
        .then(() => {
            loading.value = false;
            emits('start', selectedAlgo.value, selectedLevel.value, envWrappers)
        })
        .catch(e => {
            alert("Error while starting the training");
            console.error(e);
            loading.value = false;
        })
}

function envWrapperChanged(e: Event) {
    const target = e.target as HTMLInputElement;
    if (target.checked) {
        envWrappers.push(target.name);
    } else {
        envWrappers.splice(envWrappers.indexOf(target.name), 1);
    }
}

function algoWrapperChanged(e: Event) {
    const target = e.target as HTMLInputElement;
    if (target.checked) {
        algoWrappers.push(target.name);
    } else {
        algoWrappers.splice(algoWrappers.indexOf(target.name), 1);
    }
}

</script>