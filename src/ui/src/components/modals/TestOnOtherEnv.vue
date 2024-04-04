<template>
    <div ref="modal" class="modal fade" tabindex="-1">
        <div class="modal-dialog modal-dialog-scrollable modal-dialog-centered modal-xl">
            <div class="modal-content">
                <div class="modal-header">
                    <h5> Choose an environment from an other experiment </h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div v-show="step == 'CHOOSE_ENV'" class="modal-body row">
                    <div v-if="compatibleExperiments.length == 0">
                        <p>There is no other experiment available that has the same observation shape and extra feature
                            shape as the one you have selected.
                        </p>
                        <p>
                            Observation space: {{ originalExperiment?.env.observation_shape }}.<br>
                            Extra feature space: {{ originalExperiment?.env.extra_feature_shape }}. <br>
                            Logdir: {{ originalExperiment?.logdir }}.
                        </p>
                    </div>
                    <table v-else class="table table-sm table-striped table-hover table-scrollable">
                        <thead>
                            <tr>
                                <th class="px-1"> Logdir </th>
                                <th> Test environment </th>
                                <th></th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="exp in compatibleExperiments" @click="() => chooseEnv(exp)">
                                <td> {{ exp.logdir }} </td>
                                <td v-if="exp.test_env"> {{ exp.test_env.name }}</td>
                                <td v-else> {{ exp.env.name }}</td>
                                <td class="row">
                                    <template
                                        v-for="[seed, img] in envImages.get(exp.logdir)?.sort((a, b) => a[0] - b[0])">
                                        <div class="col">
                                            <img class="m-1" :src="'data:image/jpg;base64, ' + img" />
                                            <br />
                                            Seed = {{ seed }}
                                        </div>
                                    </template>

                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                <div v-show="step == 'CHOOSE_PARAMS'" class="modal-body row">
                    <div class="col">
                        <h5> Choose the parameters for the test </h5>
                        <div class="input-group mb-3">
                            <span class="input-group-text"> Number of tests </span>
                            <input type="number" class="form-control" v-model="nTests" />
                        </div>
                        <div class="input-group mb-3">
                            <span class="input-group-text"> New logdir </span>
                            <input type="text" class="form-control" v-model="newExperimentLogdir" />
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button v-show="step == 'CHOOSE_PARAMS'" class="btn btn-primary"
                        @click="() => step = 'CHOOSE_ENV'">Go back</button>
                    <button v-show="step == 'CHOOSE_PARAMS'" class="btn btn-success" data-bs-dismiss="modal"
                        @click="start">
                        Start
                    </button>
                    <button type="button" class="btn btn-outline-danger" data-bs-dismiss="modal">Cancel</button>
                </div>

            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { useExperimentStore } from '../../stores/ExperimentStore';
import { Modal } from 'bootstrap';
import { Experiment } from '../../models/Experiment';


const originalExperiment = ref(null as Experiment | null);
const store = useExperimentStore();
const modal = ref({} as HTMLDivElement);
const envImages = ref<Map<string, [number, string][]>>(new Map());
const step = ref("CHOOSE_ENV" as "CHOOSE_ENV" | "CHOOSE_PARAMS")
const chosenExperimentForEnv = ref(null as Experiment | null);
const nTests = ref(1);
const newExperimentLogdir = ref("");

/**
 * Compatible experiments are experiments that have 
 *  - the same observation shape
 *  - the same extra feature shape
 *  - a logdir different from the original experiment.
 */
const compatibleExperiments = computed(() => {
    if (originalExperiment.value == null) {
        return []
    }
    const logdir = originalExperiment.value.logdir;
    const observationShape = originalExperiment.value.env.observation_shape;
    const extrasShape = originalExperiment.value.env.extra_feature_shape;
    return store.experiments.filter(exp => {
        if (exp.logdir == logdir) {
            return false
        }
        // Array equality in JS is not straightforward
        if (exp.env.observation_shape.length != observationShape.length || exp.env.observation_shape.some((dim, i) => dim != observationShape[i])) {
            return false
        }
        return exp.env.extra_feature_shape.every((el, i) => extrasShape[i] == el);
    })

})

function chooseEnv(exp: Experiment) {
    chosenExperimentForEnv.value = exp;
    step.value = "CHOOSE_PARAMS";
    newExperimentLogdir.value = originalExperiment.value?.logdir + "-tested-on-" + exp.test_env?.name;
}

function start() {
    if (originalExperiment.value == null || chosenExperimentForEnv.value == null) {
        alert("Please choose an experiment")
        return
    }
    store.testOnOtherEnvironment(originalExperiment.value.logdir, newExperimentLogdir.value, chosenExperimentForEnv.value.logdir, nTests.value);
}

function showModal(experiment: Experiment) {
    originalExperiment.value = experiment;
    step.value = "CHOOSE_ENV";
    (new Modal(modal.value)).show()
    compatibleExperiments.value.forEach(exp => {
        if (!envImages.value.has(exp.logdir)) {
            envImages.value.set(exp.logdir, [])
        }
        for (let seed = 0; seed <= 1_000_000; seed += 100_000) {
            // Do not load images that are already loaded
            if (envImages.value.get(exp.logdir)!.find(([s, _]) => s == seed) != undefined) {
                continue
            }
            store.getEnvImage(exp.logdir, seed).then(img => {
                const images = envImages.value.get(exp.logdir);
                if (images == null) {
                    return
                }
                if (!images.includes([seed, img])) {
                    images.push([seed, img])
                }
            })
        }
    })
}

defineExpose({ showModal });
</script>
