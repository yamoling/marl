<template>
    <div class="row">
        <div class="col-auto">
            <div class="row mb-1">
                <h3> Train </h3>
                <div class="col-auto">
                    <div class="input-group">
                        <span class="input-group-text"> Train for </span>
                        <input type="number" class="form-control" :disabled="isBusy" v-model.number="trainSteps"
                            @keyup.enter="train" size="2" />
                        <span class="input-group-text"> steps </span>
                        <button type="button" class="btn btn-success" @click="train" :disabled="isBusy">
                            Train
                            <font-awesome-icon v-if="!isBusy" icon="fa-solid fa-solid fa-forward-step" />
                            <font-awesome-icon v-else icon="spinner" spin />
                        </button>
                    </div>
                </div>
            </div>
            <div class="row mb-1">
                <h3> Tests </h3>
                <div class="col-auto mb-3">
                    <label class="form-label">Automatic test interval</label>
                    <div class="input-group mb-1">
                        <input type="number" class="form-control" v-model.number="autoTestInterval" size="2" />
                        <span class="input-group-text"> steps </span>
                    </div>
                    <div class="input-group">
                        <span class="input-group-text"> Test </span>
                        <input type="number" class="form-control" v-model.number="numTests" size="2" />
                        <span class="input-group-text"> episodes </span>
                    </div>
                </div>
            </div>
            <div class="row">
                <h3> Checkpoints </h3>

            </div>
        </div>
        <div class="col-5">
            <div class="row">
                <MetricsPlotter title="Train metrics" :metrics="trainMetrics" :reverse-labels="true" :max-steps="50" />
            </div>
            <div class="row">
                <MetricsPlotter title="Test metrics" :metrics="testMetrics" :reverse-labels="true" :max-steps="50" />
            </div>
            <div class="row">
                <ReplayMemory />
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { HTTP_URL, wsURL } from '../../constants';
import { Metrics } from '../../models/Metric';
import { useGlobalState } from '../../stores/GlobalState';
import MetricsPlotter from '../charts/MetricsPlotter.vue';
import ReplayMemory from '../visualisation/ReplayMemory.vue';


const trainMetrics = ref([] as Metrics[]);
const testMetrics = ref([] as Metrics[]);
const autoTestInterval = ref(200);
const trainSteps = ref(500);
const numTests = ref(10);
const isBusy = ref(false);
const globalState = useGlobalState();

function train() {
    // Connect to the websocket, then start the training
    isBusy.value = true;
    const ws = new WebSocket(wsURL(globalState.wsPort || 0));
    ws.onopen = () => {
        fetch(`${HTTP_URL}/train/start`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                num_steps: trainSteps.value,
                num_tests: numTests.value,
                test_interval: autoTestInterval.value,
            })
        }).catch(e => { console.error(e); alert("Could not start training") });
    }
    ws.onclose = () => {
        isBusy.value = false;
    }
    ws.onmessage = (event: MessageEvent) => {
        const data = JSON.parse(event.data) as { step: number, tag: string, metrics: Metrics };
        if (data.tag == "Train") {
            trainMetrics.value.unshift(data.metrics);
        } else if (data.tag == "Test") {
            testMetrics.value.unshift(data.metrics);
        }
    }
}
</script>
