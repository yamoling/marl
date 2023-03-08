<template>
    <div>
        <fieldset class="row mb-2">
            <div class="col-auto">
                <legend>Experiment name</legend>
                <div class="form-check form-switch">
                    <label class="form-check-label">
                        <input v-model="autoName" class="form-check-input" type="checkbox" role="switch" />
                        Generate automatically
                    </label>
                </div>

                <div class="input-group mb-1">
                    <label class="input-group-text">Name </label>
                    <input type="text" :disabled="autoName" class="form-control" v-model="experimentName">
                </div>
            </div>
        </fieldset>

        <div class="row mb-2">
            <fieldset class="col-auto">
                <legend>Algo config</legend>
                <div class="form-check form-switch">
                    <label class="form-check-label">
                        <input v-model="isRecurrent" class="form-check-input" type="checkbox" role="switch" />
                        Recurrent
                    </label>
                </div>
                <div class="form-check form-switch">
                    <label class="form-check-label">
                        <input v-model="vdn" class="form-check-input" type="checkbox" role="switch" />
                        VDN
                    </label>
                </div>

                <div class="input-group mb-1">
                    <label class="input-group-text">Select a level</label>
                    <select class="form-select" v-model="selectedLevel">
                        <option v-for="algo in 6" :value="'lvl' + algo">
                            Level {{ algo }}
                        </option>
                    </select>
                </div>
            </fieldset>
        </div>

        <fieldset class="row mb-2">
            <div ref="envWrappers" class="col-auto">
                <legend>Env wrappers</legend>
                <div class="form-check form-switch">
                    <label class="form-check-label">
                        <input class="form-check-input" type="checkbox" role="switch" name="TimeLimit" checked />
                        TimeLimit
                        <input type="number" size="8" v-model="timeLimitValue" />
                    </label>
                </div>
                <div class="form-check form-switch">
                    <label class="form-check-label">
                        <input class="form-check-input" type="checkbox" role="switch" name="VideoRecorder" />
                        Video recorder
                    </label>
                </div>
                <div class="form-check form-switch">
                    <label class="form-check-label">
                        <input class="form-check-input" type="checkbox" role="switch" name="IntrinsicReward" />
                        Intrinsic reward
                    </label>
                </div>
                <div class="form-check form-switch">
                    <label class="form-check-label">
                        <input class="form-check-input" type="checkbox" role="switch" name="AgentId" checked />
                        Add agent ID
                    </label>
                </div>
            </div>
        </fieldset>

        <fieldset class="row mb-2">
            <div class="col-auto">
                <legend>Replay Memory</legend>
                <div class="input-group mb-1">
                    <label class="input-group-text"> Size </label>
                    <input type="text" class="form-control" v-model.number="memorySize" size="2" />
                </div>

                <div class="form-check form-switch mb-1">
                    <label class="form-check-label">
                        <input class="form-check-input" type="checkbox" role="switch" name="AgentId"
                            v-model="prioritizedMemory" />
                        Prioritized
                    </label>
                </div>

                <div class="input-group mb-1">
                    <label class="input-group-text"> N step return </label>
                    <input type="text" class="form-control" v-model.number="nStep" size="2" />
                </div>
            </div>
        </fieldset>

        <button v-if="!loading" role="button" class="btn btn-primary" @click="send">
            Start
            <font-awesome-icon icon="fa-solid fa-play" />
        </button>
        <button v-else role="button" class="btn btn-primary" disabled>
            <font-awesome-icon icon="spinner" spin />
        </button>
    </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { HTTP_URL } from '../../constants';
import { useGlobalState } from '../../stores/GlobalState';
import { useReplayStore } from '../../stores/ReplayStore';

const globalState = useGlobalState();
const loading = ref(false);
const customExperimentName = ref("");
const autoName = ref(true);
const isRecurrent = ref(false);
const vdn = ref(true);
const selectedLevel = ref("lvl3");
const timeLimitValue = ref(20);
const nStep = ref(1);
const memorySize = ref(10000);
const prioritizedMemory = ref(false);
const envWrappers = ref({} as HTMLElement);
const emits = defineEmits(['start']);

const experimentName = computed(() => {
    if (!autoName.value) {
        if (!customExperimentName.value.startsWith("logs/")) {
            customExperimentName.value = `logs/${customExperimentName.value}`;
        }
        return customExperimentName.value;
    }
    return computeAutoName();
});

function send() {
    loading.value = true;
    fetch(`${HTTP_URL}/train/create`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            recurrent: isRecurrent.value,
            logdir: experimentName.value,
            vdn: vdn.value,
            env_wrappers: gatherSelectedEnvWrappers(),
            time_limit: timeLimitValue.value,
            level: selectedLevel.value,
            memory: {
                size: memorySize.value,
                prioritized: prioritizedMemory.value,
                nstep: nStep.value
            },
        })
    })
        .then(resp => resp.json())
        .then((data: { logdir: string, port: number }) => {
            const replayStore = useReplayStore();
            globalState.logdir = data.logdir;
            globalState.wsPort = data.port;
            replayStore.logdirs.push(data.logdir);
            loading.value = false;
            emits('start', isRecurrent.value, vdn.value, selectedLevel.value, envWrappers)
        })
        .catch(e => {
            alert("Error while starting the training");
            console.error(e);
            loading.value = false;
        })
}

function gatherSelectedEnvWrappers() {
    return [...envWrappers.value.querySelectorAll("input[type=checkbox]:checked")]
        .map(i => {
            const input = i as HTMLInputElement;
            return input.name;
        });
}

function computeAutoName() {
    let name = `logs/${selectedLevel.value}`;
    if (vdn.value) {
        name += "-vdn";
        if (isRecurrent.value) {
            name += "-recurrent";
        }
    } else if (isRecurrent.value) {
        name += "-rdqn";
    } else {
        name += "-dqn";
    }
    if (prioritizedMemory.value) {
        name += "-per";
    }
    if (nStep.value > 1) {
        name += `-${nStep.value}_steps`;
    }
    const now = new Date();
    name += `-${now.toISOString()}`
    customExperimentName.value = name;
    return name;
}

</script>