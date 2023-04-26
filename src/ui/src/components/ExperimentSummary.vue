<template>
    <div class="row">
        <span class="badge rounded-pill text-bg-success col-auto me-1" v-for="env in envBadges"> {{ env }}</span>
        <span class="badge rounded-pill text-bg-secondary col-auto me-1" v-for="wrapper in experiment.env.wrappers"> {{
            wrapper }}</span>
        <span class="badge rounded-pill text-bg-warning col-auto me-1" v-for="algo in algoBadges"> {{ algo }} </span>
        <span class="badge rounded-pill text-bg-info col-auto" style="cursor: pointer;" @click="showParameters"> See all
        </span>
        <ExperimentParameters id="paramsModal" :experiment="experiment" />
    </div>
</template>

<script setup lang="ts">
import { Modal } from 'bootstrap';
import { computed } from 'vue';
import { ExperimentInfo } from '../models/Infos';
import { DQNInfo } from "../models/Algos";
import ExperimentParameters from './modals/ExperimentParameters.vue';


const props = defineProps<{
    experiment: ExperimentInfo;
}>();
const envBadges = computed(() => {
    const obsType = props.experiment.env.DynamicLaserEnv?.obs_type || props.experiment.env.StaticLaserEnv?.obs_type;
    return [
        `${props.experiment.env.name}(${props.experiment.env.n_agents})`,
        obsType + ": " + props.experiment.env.obs_shape,
        "Extras: " + props.experiment.env.extras_shape,
    ]
});
const algoBadges = computed(() => {
    const res = [] as string[];
    if (props.experiment.algorithm.qnetwork != null) {
        const algo = props.experiment.algorithm as DQNInfo;
        res.concat([
            `${algo.name}(${algo.qnetwork.name})`,
            "Train: " + algo.train_policy.name,
            "Test: " + algo.test_policy.name,
        ])
    }
    return res;
});

function showParameters() {
    const modal = new Modal("#paramsModal");
    modal.show();
}

</script>