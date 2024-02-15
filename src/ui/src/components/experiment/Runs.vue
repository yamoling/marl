<template>
    <div>
        <h3 class="text-center"> Runs </h3>
        <div class="table-scrollable">
            <table class="table table-sm-table-striped table-borderless">
                <tbody>
                    <template v-for="(run, i) in experiment.runs">
                        <tr style="border-top: 1px solid black;">
                            <td> {{ run.rundir.split('/').at(-1) }} </td>
                            <td v-if="runStatus[i] == 'paused'"> 💤 </td>
                            <td v-else-if="runStatus[i] == 'running'">
                                <font-awesome-icon :icon="['fas', 'spinner']" spin />
                            </td>
                            <td v-else>
                                <font-awesome-icon :icon="['fas', 'check']" />
                            </td>
                            <td>
                                <button v-if="runStatus[i] == 'paused'" class="btn btn-sm btn-outline-success"
                                    @click="() => restartRun(i, run)">
                                    <font-awesome-icon :icon="['fas', 'play']" />
                                </button>
                                <button v-else-if="runStatus[i] == 'running'" class="btn btn-sm btn-outline-warning"
                                    @click="() => pauseRun(i, run.rundir)" :disabled="pausing[i]">
                                    <font-awesome-icon v-if="!pausing[i]" :icon="['fas', 'pause']" />
                                    <font-awesome-icon v-else :icon="['fas', 'spinner']" spin />
                                </button>
                                <button class="ms-1 btn btn-sm btn-outline-danger" @click="() => deleteRun(i, run)"
                                    :disabled="deleting[i]">
                                    <font-awesome-icon v-if="deleting[i]" :icon="['fas', 'spinner']" spin />
                                    <font-awesome-icon v-else :icon="['fas', 'trash']" />
                                </button>
                            </td>
                        </tr>
                        <tr>
                            <td colspan="3">
                                <div class="progress position-relative" role="progressbar">
                                    <div class="progress-bar bg-info progress-bar-striped text-dark"
                                        :class="(run.port != null ? 'progress-bar-animated' : '')" role="progressbar"
                                        :style="{ width: `${100 * (progresses.get(run.rundir) || 0) / experiment.n_steps}%` }">
                                    </div>
                                    <div class="justify-content-center d-flex position-absolute w-100">
                                        {{ progresses.get(run.rundir) }} / {{ experiment.n_steps }}
                                        ({{ (100 * (progresses.get(run.rundir) || 0) / experiment.n_steps).toFixed(2) }}%)
                                    </div>
                                </div>
                            </td>
                        </tr>
                    </template>
                    <tr>
                        <td colspan="3" class="text-center">
                            <button class="btn btn-sm btn-outline-success ms-4 px-3" @click="createNewRunner">
                                <font-awesome-icon :icon="['fas', 'plus']" />
                            </button>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</template>
<script setup lang="ts">
import { Modal } from 'bootstrap';
import { computed, onMounted, ref, watch } from 'vue';
import { ReplayEpisodeSummary } from '../../models/Episode';
import { Experiment } from '../../models/Experiment';
import { Run } from '../../models/Experiment';
import { useRunStore } from '../../stores/RunStore';

let runConfigModal = {} as Modal;
let restartRunModal = {} as Modal;
const runToRestart = ref(null as null | Run);
const store = useRunStore();
// Map rundir to run.time_step.
const progresses = ref(new Map<string, number>());
const runStatus = computed(() => {
    // Status is either paused, running or completed.
    return props.experiment.runs.map(r => {
        if (progresses.value.get(r.rundir) == props.experiment.n_steps) {
            return 'completed';
        }
        if (r.port != null) {
            return 'running';
        }
        return 'paused';
    });
});
const props = defineProps<{
    experiment: Experiment
}>();
const deleting = ref(props.experiment.runs.map(_ => false));
const pausing = ref(props.experiment.runs.map(_ => false));



async function deleteRun(index: number, run: Run) {
    if (confirm(`Are you sure you want to delete ${run.rundir}?`)) {
        deleting.value[index] = true;
        try {
            await store.deleteRun(run.rundir);
            props.experiment.runs.splice(index, 1);
        } catch (e) {
            alert(`Failed to delete ${run.rundir}: ${e}`);
        }
        deleting.value[index] = false;
    }
}

async function restartRun(index: number, run: Run) {
    runToRestart.value = run;
    restartRunModal.show()
}

async function onRunRestart() {
    restartRunModal.hide();
    let port = await store.getRunnerPort(runToRestart.value!.rundir);
    while (port == null) {
        port = await store.getRunnerPort(runToRestart.value!.rundir);
    }
    emits("reload-requested")
}

function onTrainUpdate(rundir: string, data: ReplayEpisodeSummary) {
    const step = Number.parseInt(data.name);
    progresses.value.set(rundir, step);
}

async function pauseRun(runNum: number, rundir: string) {
    if (confirm(`Are you sure you want to pause ${rundir}?`)) {
        pausing.value[runNum] = true;
        await store.stopRunner(rundir);
        pausing.value[runNum] = false;

    }
}


async function createNewRunner() {
    store.createRunner({
        logdir: props.experiment.logdir,
        num_tests: 5,
        seed: null
    });
    return
    // const runConfig = {
    //     checkpoint: null,
    //     logdir: props.experiment.logdir,
    //     num_runs: nRuns.value,
    //     num_tests: nTests.value,
    //     test_interval: testInterval.value,
    //     num_steps: nSteps.value,
    //     use_seed: useSeed.value,
    // } as RunConfig;
    // store.createRunner(runConfig)
}


const emits = defineEmits<{
    (event: "new-test", rundir: string, data: ReplayEpisodeSummary): void
    (event: "reload-requested"): void
}>();

</script>

<style scoped>
table td:nth-child(2),
table td:nth-child(3) {
    text-align: center;
}
</style>../stores/RunStore