<template>
    <div>
        <RunConfig :id="runConfigModalID" :experiment="experiment" @run-started="onRunStart" />
        <RunRestart :id="restartRunModalID" :run="runToRestart" @run-restarted="onRunRestart" />
        <h3 class="text-center">
            Runs
            <button class="btn btn-sm btn-success ms-4 px-3" @click="() => runConfigModal.show()">
                <font-awesome-icon :icon="['fas', 'plus']" />
            </button>
        </h3>
        <div class="table-scrollable">
            <table class="table table-sm-table-striped table-borderless">
                <thead>
                    <tr>
                        <th> Rundir </th>
                        <th class="text-center"> Status </th>
                        <th> </th>
                    </tr>
                </thead>
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
                                        :style="{ width: `${progresses.get(run.rundir)}%` }">
                                    </div>
                                    <div class="justify-content-center d-flex position-absolute w-100"> {{
                                        progresses.get(run.rundir)?.toFixed(2) }}% </div>
                                </div>
                            </td>
                        </tr>
                    </template>
                </tbody>
            </table>
        </div>
    </div>
</template>
<script setup lang="ts">
import { Modal } from 'bootstrap';
import { computed, onMounted, ref, watch } from 'vue';
import { ReplayEpisodeSummary } from '../models/Episode';
import { Experiment } from '../models/Experiment';
import { RunInfo } from '../models/Infos';
import { useRunnerStore } from '../stores/RunnerStore';
import RunConfig from './modals/RunConfig.vue';
import RunRestart from './modals/RunRestart.vue';

let runConfigModal = {} as Modal;
let restartRunModal = {} as Modal;
const runToRestart = ref(null as null | RunInfo);
const store = useRunnerStore();
// Map rundir to progress (from 0 to 100%).
const progresses = ref(new Map<string, number>());
const numSteps = new Map<string, number>();
const runStatus = computed(() => {
    // Status is either paused, running or completed.
    return props.experiment.runs.map(r => {
        if (progresses.value.get(r.rundir) == 100) {
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
const runConfigModalID = `${props.experiment.logdir}-runConfigModal`;
const restartRunModalID = `${props.experiment.logdir}-restartRunModal`;


onMounted(() => {
    runConfigModal = new Modal("#" + runConfigModalID);
    restartRunModal = new Modal("#" + restartRunModalID);
    updateListeners(props.experiment.runs);
})

function updateListeners(runs: RunInfo[]) {
    runs.forEach(run => {
        progresses.value.set(run.rundir, run.current_step * 100 / run.stop_step);
        if (run.pid != null && run.port != null) {
            store.startListening(
                run.rundir,
                run.port,
                (data: ReplayEpisodeSummary) => onTrainUpdate(run.rundir, data),
                (data: ReplayEpisodeSummary) => onTestUpdate(run.rundir, data),
                () => onClose(run.rundir)
            );
        }
        numSteps.set(run.rundir, run.stop_step);
    });
}


watch(() => props.experiment.runs, updateListeners);


async function deleteRun(index: number, run: RunInfo) {
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

async function restartRun(index: number, run: RunInfo) {
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
    const numStep = numSteps.get(rundir);
    if (numStep != null) {
        progresses.value.set(rundir, 100 * step / numStep);
    }
}

async function pauseRun(runNum: number, rundir: string) {
    if (confirm(`Are you sure you want to pause ${rundir}?`)) {
        pausing.value[runNum] = true;
        await store.stopRunner(rundir);
        pausing.value[runNum] = false;

    }
}


function onTestUpdate(rundir: string, data: ReplayEpisodeSummary) {
    emits("new-test", rundir, data);
}

function onClose(rundir: string) {
    // Set the port to null
    props.experiment.runs.forEach(run => {
        if (run.rundir == rundir) {
            run.port = null;
        }
    });
}

function onRunStart() {
    runConfigModal.hide();
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
</style>