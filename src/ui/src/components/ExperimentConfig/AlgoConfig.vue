<template>
    <fieldset>
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
        <div class="form-check form-switch">
            <label class="form-check-label">
                <input class="form-check-input" type="checkbox" role="switch" v-model="forceActions" />
                Force actions
                <ul>
                    <li v-for="[agent, action] in forcedActions">
                        <button class="btn btn-sm btn-outline-danger" @click.stop="() => forcedActions.delete(agent)">
                            <font-awesome-icon icon="fa fa-minus" />
                        </button>
                        Agent {{ agent }}: {{ ACTION_MEANINGS[action] }}
                    </li>
                    <li class="input-group">
                        <label class="input-group-text"> Agent </label>
                        <select class="form-select" style="width: 80px;" v-model.number="forcedAgent">
                            <option v-for="agent in 4" :value="agent - 1"> {{ agent - 1 }} </option>
                        </select>
                        <label class="input-group-text"> Action </label>
                        <select class="form-select" style="width: 100px;" v-model.number="forcedAction">
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
        <div class="input-group">
            <label class="input-group-text"> Train Policy </label>
            <select class="form-select" v-model="trainPolicy">
                <option v-for="policy in POLICIES" :value="policy"> {{ policy }} </option>
            </select>
        </div>
        <div v-if="trainPolicy == 'DecreasingEpsilonGreedyPolicy'" class="ms-5 my-1">
            <div class="input-group">
                <label class="input-group-text"> Epsilon </label>
                <input type="text" class="form-control" v-model.number="epsilon" size="2" />
            </div>
            <div class="input-group">
                <label class="input-group-text"> Anneal on </label>
                <input type="text" class="form-control" v-model.number="annealOn" size="2" />
                <span class="input-group-text"> steps </span>
            </div>
        </div>
        <div class="input-group">
            <label class="input-group-text"> Train Policy </label>
            <select class="form-select" v-model="testPolicy">
                <option v-for="policy in POLICIES" :value="policy"> {{ policy }} </option>
            </select>
        </div>
    </fieldset>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { ACTION_MEANINGS, POLICIES } from "../../constants";

// Algo config
const isRecurrent = ref(false);
const vdn = ref(true);
const forceActions = ref(false);
const forcedActions = ref(new Map<number, number>());
const forcedAgent = ref(0);
const forcedAction = ref(0);
const trainPolicy = ref('DecreasingEpsilonGreedyPolicy' as typeof POLICIES[number]);
const testPolicy = ref('ArgMax' as typeof POLICIES[number]);
const epsilon = ref(1);
const annealOn = ref(10_000);


function addForcedAgent() {
    forcedActions.value.set(forcedAgent.value, forcedAction.value);
    forcedAction.value = 0;
    forcedAgent.value += 1;
}
</script>