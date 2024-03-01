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
                                    :disabled="pausing[i]">
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
                            <button class="btn btn-sm btn-outline-success ms-4 px-3">
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
import { computed, ref } from 'vue';
import { ReplayEpisodeSummary } from '../../models/Episode';
import { Experiment } from '../../models/Experiment';
import { Run } from '../../models/Experiment';

let restartRunModal = {} as Modal;
const runToRestart = ref(null as null | Run);
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
            alert("Not implemented !");
            return
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

function onTrainUpdate(rundir: string, data: ReplayEpisodeSummary) {
    const step = Number.parseInt(data.name);
    progresses.value.set(rundir, step);
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