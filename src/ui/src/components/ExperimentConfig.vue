<template>
    <div ref="modal" class="modal fade" tabindex="-1">
        <div class="modal-dialog modal-dialog-centered modal-xl modal-dialog-scrollable ">
            <div class="modal-content">
                <div class="modal-header">
                    <h5> Experiment creation </h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body row">
                    <div class="col-auto mx-auto">
                        <fieldset>
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
                        </fieldset>

                        <div class="row mb-2">
                            <fieldset class="col-auto">
                                <legend>Algo config</legend>
                                <div class="form-check form-switch">
                                    <label class="form-check-label">
                                        <input v-model="isRecurrent" class="form-check-input" type="checkbox"
                                            role="switch" />
                                        Recurrent
                                    </label>
                                </div>
                                <div class="form-check form-switch">
                                    <label class="form-check-label">
                                        <input v-model="vdn" class="form-check-input" type="checkbox" role="switch" />
                                        VDN
                                    </label>
                                </div>
                            </fieldset>
                        </div>
                        <fieldset class="row mb-2">
                            <div ref="envWrappers" class="col-auto">
                                <legend>Env wrappers</legend>
                                <div class="form-check form-switch">
                                    <label class="form-check-label">
                                        <input class="form-check-input" type="checkbox" role="switch" name="TimeLimit"
                                            checked />
                                        TimeLimit
                                        <input type="number" size="8" v-model.number="timeLimitValue" />
                                    </label>
                                </div>
                                <div class="form-check form-switch">
                                    <label class="form-check-label">
                                        <input class="form-check-input" type="checkbox" role="switch"
                                            name="IntrinsicReward" />
                                        Intrinsic reward
                                    </label>
                                </div>
                                <div class="form-check form-switch">
                                    <label class="form-check-label">
                                        <input class="form-check-input" type="checkbox" role="switch" name="AgentId"
                                            checked />
                                        Add agent ID
                                    </label>
                                </div>
                                <div class="form-check form-switch">
                                    <label class="form-check-label">
                                        <input class="form-check-input" type="checkbox" role="switch" name="Penalty" />
                                        Time penalty
                                        <input type="number" size="8" v-model.number="timePenalty" />
                                    </label>
                                </div>
                                <div class="form-check form-switch">
                                    <label class="form-check-label">
                                        <input class="form-check-input" type="checkbox" role="switch" name="ForceAction" />
                                        Force actions
                                        <ul>
                                            <li v-for="[agent, action] in forcedActions">
                                                <button class="btn btn-sm btn-outline-danger"
                                                    @click.stop="() => forcedActions.delete(agent)">
                                                    <font-awesome-icon icon="fa fa-minus" />
                                                </button>
                                                Agent {{ agent }}: {{ ACTION_MEANINGS[action] }}
                                            </li>
                                            <li class="input-group">
                                                <label class="input-group-text"> Agent </label>
                                                <select class="form-select" style="width: 80px;"
                                                    v-model.number="forcedAgent">
                                                    <option v-for="agent in 4" :value="agent - 1"> {{ agent - 1 }} </option>
                                                </select>
                                                <label class="input-group-text"> Action </label>
                                                <select class="form-select" style="width: 100px;"
                                                    v-model.number="forcedAction">
                                                    <option v-for="(action, value) in ACTION_MEANINGS" :value="value"> {{
                                                        action
                                                    }} </option>
                                                </select>
                                                <button class="btn btn-sm btn-outline-success" @click="addForcedAgent">
                                                    <font-awesome-icon icon="fa fa-plus" />
                                                </button>
                                            </li>
                                        </ul>
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
                    </div>
                    <div class="col-auto mx-auto">
                        <fieldset>
                            <legend> Map config </legend>
                            <div class="form-check form-switch">
                                <label class="form-check-label">
                                    <input class="form-check-input" type="checkbox" role="switch" v-model="staticMap" />
                                    Static map
                                </label>
                            </div>
                            <div class="input-group mb-3">
                                <label class="input-group-text">Observation type</label>
                                <select class="form-select" v-model="obsType">
                                    <option v-for="obs in OBS_TYPES" :value="obs"> {{ obs }}</option>
                                </select>
                            </div>



                            <div v-if="staticMap">
                                <div v-for="col in 2" class="row">
                                    <div v-for="level in 3" class="col-auto form-check">
                                        <label class="form-check-label bg-body-tertiary rounded p-2"
                                            :class="(selectedLevel == (level + (col - 1) * 3)) ? 'bg-success-subtle' : ''"
                                            @click="() => selectedLevel = level + (col - 1) * 3">
                                            <img :src="`img/levels/lvl${level + (col - 1) * 3}.png`"
                                                style="{width: 100px; height: 100px;}">
                                            Level {{ level + (col - 1) * 3 }}
                                        </label>
                                    </div>
                                </div>
                            </div>
                            <div v-else>
                                <legend>Auto-generated maps</legend>
                                <div class="row mb-1">
                                    <h6 class="col-auto my-auto">
                                        Map size
                                    </h6>
                                    <div class="col-auto">
                                        <div class="input-group">
                                            <span class="input-group-text"> Width </span>
                                            <input type="number" class="form-control" v-model.number="width" size="2">
                                        </div>
                                        <div class="input-group">
                                            <span class="input-group-text"> Height </span>
                                            <input type="number" class="form-control" v-model.number="height" size="2">
                                        </div>
                                    </div>
                                </div>
                                <div class="input-group mb-1">
                                    <label class="input-group-text"> Number of agents </label>
                                    <input type="number" class="form-control" v-model.number="numAgents" size="2">
                                </div>
                                <div class="input-group mb-1">
                                    <label class="input-group-text"> Number of lasers </label>
                                    <input type="number" class="form-control" v-model.number="numLasers" size="2">
                                </div>
                                <div class="input-group mb-1">
                                    <label class="input-group-text"> Number of gems </label>
                                    <input type="number" class="form-control" v-model.number="numGems" size="2">
                                </div>
                                <div class="input-group mb-1">
                                    <label class="input-group-text"> Wall density </label>
                                    <input type="text" class="form-control" v-model.number="wallDensity" size="2">
                                    <input type="range" min="0" max="1" step="0.01" class="form-range"
                                        v-model="wallDensity">
                                </div>
                            </div>
                        </fieldset>
                    </div>
                </div>
                <div class="modal-footer">
                    <button role="button" class="btn btn-success" @click="send">
                        <font-awesome-icon v-if="loading" icon="spinner" spin />
                        <span v-else> Start
                            <font-awesome-icon icon="fa-solid fa-play" />
                        </span>
                    </button>
                    <button type="button" class="btn btn-outline-danger" data-bs-dismiss="modal">Cancel</button>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { OBS_TYPES } from "../models/Infos";
import { useExperimentStore } from '../stores/ExperimentStore';
import { ACTION_MEANINGS } from "../constants";

const experimentStore = useExperimentStore();
const loading = ref(false);

// Algo config
const customExperimentName = ref("");
const autoName = ref(true);
const isRecurrent = ref(false);
const vdn = ref(true);

// Map config
const obsType = ref("FLATTENED" as typeof OBS_TYPES[number]);
const staticMap = ref(true);
const selectedLevel = ref(3);
const width = ref(5);
const height = ref(5);
const numAgents = ref(2);
const numLasers = ref(0);
const numGems = ref(3);
const wallDensity = ref(0.2);

// Env wrappers
const timeLimitValue = ref(20);
const nStep = ref(1);
const memorySize = ref(10000);
const prioritizedMemory = ref(false);
const envWrappers = ref({} as HTMLElement);
const timePenalty = ref(-0.1);
const forcedActions = ref(new Map<number, number>());
const forcedAgent = ref(0);
const forcedAction = ref(0);
const emits = defineEmits(["experiment-created"]);

const experimentName = computed({
    get() {
        if (!autoName.value) {
            if (!customExperimentName.value.startsWith("logs/")) {
                customExperimentName.value = `logs/${customExperimentName.value}`;
            }
            return customExperimentName.value;
        }
        return computeAutoName();
    },
    set(newValue) {
        customExperimentName.value = newValue;
    },
});

async function send() {
    loading.value = true;
    try {
        const logdir = await experimentStore.createExperiment(experimentName.value, {
            recurrent: isRecurrent.value,
            logdir: experimentName.value,
            vdn: vdn.value,
            env_wrappers: gatherSelectedEnvWrappers(),
            time_limit: timeLimitValue.value,
            static_map: staticMap.value,
            level: `lvl${selectedLevel.value}`,
            forced_actions: Object.fromEntries(forcedActions.value),
            obs_type: obsType.value,
            time_penalty: timePenalty.value,
            generator: {
                width: width.value,
                height: height.value,
                n_agents: numAgents.value,
                n_lasers: numLasers.value,
                n_gems: numGems.value,
                wall_density: wallDensity.value,
            },
            memory: {
                size: memorySize.value,
                prioritized: prioritizedMemory.value,
                nstep: nStep.value
            },
        });
        loading.value = false;
        emits("experiment-created", logdir);
    }
    catch (e) {
        alert("Error while starting the training");
        console.error(e);
        loading.value = false;
    }
}

function gatherSelectedEnvWrappers() {
    return [...envWrappers.value.querySelectorAll("input[type=checkbox]:checked")]
        .map(i => {
            const input = i as HTMLInputElement;
            return input.name;
        });
}

function addForcedAgent() {
    forcedActions.value.set(forcedAgent.value, forcedAction.value);
    forcedAction.value = 0;
    forcedAgent.value += 1;
}

function computeAutoName() {
    let name = `logs/lvl${selectedLevel.value}`;
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