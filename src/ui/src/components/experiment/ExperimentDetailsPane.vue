<template>
    <aside class="details-pane" :class="{ collapsed: !isOpen }">
        <button class="btn btn-outline-secondary btn-sm pane-toggle" type="button" @click="emits('toggle')">
            <font-awesome-icon :icon="isOpen ? 'fa-solid fa-chevron-left' : 'fa-solid fa-chevron-right'" />
        </button>

        <div v-if="isOpen" class="details-content">
            <JsonInspector class="mb-2" :title="experiment.trainer.name" :value="experiment.trainer" />
            <JsonInspector :title="experiment.env.name" :value="experiment.env" />
        </div>
    </aside>
</template>

<script setup lang="ts">
import { Experiment } from '../../models/Experiment';
import JsonInspector from './JsonInspector.vue';

defineProps<{
    experiment: Experiment,
    isOpen: boolean
}>();

const emits = defineEmits<{
    (event: 'toggle'): void
}>();
</script>

<style scoped>
.details-pane {
    min-width: 300px;
    border: 1px solid var(--bs-border-color);
    border-radius: 0.5rem;
    background: var(--bs-body-bg);
    padding: 0.5rem;
    transition: width 0.2s ease;
    position: relative;
    overflow: auto;
    flex-shrink: 0;
}

.details-pane.collapsed {
    width: 44px;
    padding: 0.5rem 0.25rem;
}

.pane-toggle {
    width: 32px;
    height: 32px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}

.details-content {
    margin-top: 0.75rem;
}
</style>
