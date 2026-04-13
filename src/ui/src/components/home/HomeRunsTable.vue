<template>
    <div class="ps-3">
        <h5 class="mb-3">Runs</h5>
        <div v-if="runs.length === 0" class="text-muted">
            No runs found.
        </div>
        <DataTable v-else :value="runs" dataKey="rundir" striped-rows size="small">
            <Column style="width: 4rem">
                <template #header>
                    Status
                </template>
                <template #body="{ data }">
                    <font-awesome-icon :icon="statusIcon(data.status)" :class="statusClass(data.status)"
                        :title="statusLabel(data.status)" :aria-label="statusLabel(data.status)" />
                </template>
            </Column>
            <Column style="min-width: 12rem">
                <template #header>
                    Run
                </template>
                <template #body="{ data }">
                    {{ data.rundir.split('/').at(-1) }}
                </template>
            </Column>
            <Column style="min-width: 12rem">
                <template #header>
                    Progress
                </template>
                <template #body="{ data }">
                    <div class="progress position-relative" role="progressbar">
                        <div class="progress-bar text-dark" :class="progressBarClass(data)"
                            :style="{ width: `${progressPercent(data)}%` }">
                        </div>
                        <div class="justify-content-center d-flex position-absolute w-100">
                            {{ progressPercent(data).toFixed(1) }}%
                        </div>
                    </div>
                </template>
            </Column>
            <Column style="width: 8rem">
                <template #body="{ data }">
                    <button v-if="data.status === 'RUNNING'" class="btn btn-sm btn-outline-danger"
                        @click.stop="emit('stop-run', data.rundir)" :disabled="!!stoppingRuns[data.rundir]">
                        <font-awesome-icon v-if="stoppingRuns[data.rundir]" :icon="['fas', 'spinner']" spin />
                        <font-awesome-icon v-else :icon="['fas', 'stop']" />
                    </button>
                    <button v-else-if="data.status === 'CREATED'" class="btn btn-sm btn-outline-primary"
                        @click.stop="emit('start-run', data.rundir)" :disabled="!!startingRuns[data.rundir]">
                        <font-awesome-icon v-if="startingRuns[data.rundir]" :icon="['fas', 'spinner']" spin />
                        <font-awesome-icon v-else :icon="['fas', 'play']" />
                    </button>
                    <button v-else-if="data.status === 'CANCELLED'" class="btn btn-sm btn-outline-primary"
                        @click.stop="emit('start-run', data.rundir)" :disabled="!!startingRuns[data.rundir]">
                        <font-awesome-icon v-if="startingRuns[data.rundir]" :icon="['fas', 'spinner']" spin />
                        <font-awesome-icon v-else :icon="['fas', 'repeat']" />
                    </button>
                    <font-awesome-icon v-else :icon="['fas', 'check']" class="text-success" />
                </template>
            </Column>
        </DataTable>
    </div>
</template>

<script setup lang="ts">
import { DataTable, Column } from 'primevue';
import { Run, RunStatus } from '../../models/Run';

defineProps<{
    runs: Run[]
    startingRuns: Record<string, boolean>
    stoppingRuns: Record<string, boolean>
}>();

const emit = defineEmits<{
    (event: 'start-run', rundir: string): void
    (event: 'stop-run', rundir: string): void
}>();

function progressPercent(run: Run) {
    switch (run.status) {
        case 'CREATED':
            return 0;
        case 'COMPLETED':
            return 100;
        case 'RUNNING':
        case 'CANCELLED':
            return Math.min(100, run.progress * 100);
    }
}

function progressBarClass(run: Run) {
    const classes: Record<RunStatus, string> = {
        CREATED: 'bg-light',
        RUNNING: 'bg-info progress-bar-striped progress-bar-animated',
        COMPLETED: 'bg-success',
        CANCELLED: 'bg-warning',
    };
    return classes[run.status];
}

function statusIcon(status: RunStatus) {
    const icons: Record<RunStatus, ['fas', string]> = {
        CREATED: ['fas', 'clock'],
        RUNNING: ['fas', 'spinner'],
        CANCELLED: ['fas', 'ban'],
        COMPLETED: ['fas', 'check-circle'],
    };
    return icons[status];
}

function statusClass(status: RunStatus) {
    const classes: Record<RunStatus, string> = {
        CREATED: 'text-secondary',
        RUNNING: 'text-secondary fa-spin',
        CANCELLED: 'text-secondary',
        COMPLETED: 'text-success',
    };
    return classes[status];
}

function statusLabel(status: RunStatus) {
    const labels: Record<RunStatus, string> = {
        CREATED: 'Created',
        RUNNING: 'Running',
        CANCELLED: 'Cancelled',
        COMPLETED: 'Completed',
    };
    return labels[status];
}
</script>
