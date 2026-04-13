<template>
    <section class="action-panel">
        <div class="action-panel-header">
            <h6 class="mb-1">Action Visualizer</h6>
            <span class="badge text-bg-light border text-uppercase">{{ resolvedKind }}</span>
        </div>

        <div class="agent-selector" v-if="nAgents > 1">
            <button v-for="agent in availableAgents" :key="agent" type="button" class="agent-chip"
                :class="{ selected: selectedAgentsSet.has(agent) }" @click="toggleAgent(agent)">
                A{{ agent + 1 }}
            </button>
        </div>

        <DiscreteActionMatrix v-if="resolvedKind === 'discrete'" :episode="episode" :current-step="safeStep"
            :selected-agents="selectedAgents" :action-space="discreteActionSpace" />

        <ContinuousActionRadar v-else :episode="episode" :current-step="safeStep" :selected-agents="selectedAgents"
            :action-space="continuousActionSpace" />
    </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { ActionSpace, ContinuousActionSpace, DiscreteActionSpace } from '../../../models/Env';
import { ReplayEpisode } from '../../../models/Episode';
import DiscreteActionMatrix from './DiscreteActionMatrix.vue';
import ContinuousActionRadar from './ContinuousActionRadar.vue';

const props = defineProps<{
    episode: ReplayEpisode
    currentStep: number
    actionSpace: ActionSpace
    nAgents: number
}>();

const selectedAgents = ref<number[]>([]);

const maxStep = computed(() => Math.max(0, props.episode.episode.actions.length - 1));
const safeStep = computed(() => Math.max(0, Math.min(maxStep.value, props.currentStep)));
const availableAgents = computed(() => Array.from({ length: props.nAgents }, (_, index) => index));
const selectedAgentsSet = computed(() => new Set(selectedAgents.value));

const resolvedKind = computed<'discrete' | 'continuous'>(() => {
    if (props.actionSpace.space_type === 'continuous') return 'continuous';
    return 'discrete';
});

const discreteActionSpace = computed<DiscreteActionSpace>(() => {
    if (resolvedKind.value === 'discrete') {
        return {
            ...props.actionSpace,
            labels: props.actionSpace.labels ?? [],
            space_type: 'discrete',
        };
    }
    return {
        shape: props.actionSpace.shape,
        labels: [],
        space_type: 'discrete',
    };
});

const continuousActionSpace = computed<ContinuousActionSpace>(() => {
    if (resolvedKind.value === 'continuous') {
        return {
            ...props.actionSpace,
            space_type: 'continuous',
        };
    }
    return {
        shape: props.actionSpace.shape,
        space_type: 'continuous',
        low: undefined,
        high: undefined,
    };
});


watch(
    () => props.nAgents,
    () => {
        selectedAgents.value = Array.from({ length: props.nAgents }, (_, index) => index);
    },
    { immediate: true }
);

function toggleAgent(agent: number) {
    const selected = new Set(selectedAgents.value);
    if (selected.has(agent)) {
        if (selected.size === 1) return;
        selected.delete(agent);
    } else {
        selected.add(agent);
    }
    selectedAgents.value = Array.from(selected).sort((a, b) => a - b);
}
</script>

<style scoped>
.action-panel {
    margin-top: 0.55rem;
    border: 1px solid var(--bs-border-color);
    border-radius: 0.5rem;
    background: var(--bs-secondary-bg);
    padding: 0.6rem;
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
}

.action-panel-header {
    display: flex;
    justify-content: space-between;
    align-items: start;
    gap: 0.5rem;
}

.action-panel-header h6 {
    margin: 0;
}

.action-panel-header p {
    font-size: 0.78rem;
    line-height: 1.2;
}

.agent-selector {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
}

.agent-chip {
    border: 1px solid var(--bs-border-color);
    background: var(--bs-body-bg);
    border-radius: 999px;
    padding: 0.15rem 0.55rem;
    font-size: 0.74rem;
    color: var(--bs-body-color);
}

.agent-chip.selected {
    border-color: var(--bs-primary);
    background: color-mix(in srgb, var(--bs-primary) 18%, var(--bs-body-bg));
    font-weight: 700;
}
</style>
