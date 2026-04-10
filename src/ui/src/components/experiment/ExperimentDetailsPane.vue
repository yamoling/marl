<template>
    <aside class="details-pane" :class="{ collapsed: !isOpen }">
        <button class="btn btn-outline-secondary btn-sm pane-toggle" type="button" @click="emits('toggle')">
            <font-awesome-icon :icon="isOpen ? 'fa-solid fa-chevron-left' : 'fa-solid fa-chevron-right'" />
        </button>

        <div v-if="isOpen" class="details-content">
            <h5 class="mb-3">Inspection Details</h5>

            <div class="placeholder-card mb-2">
                <h6 class="mb-1">Trainer</h6>
                <p class="mb-0">{{ experiment.trainer.name }}</p>
                <small class="text-muted">Trainer details placeholder</small>
            </div>

            <div class="placeholder-card">
                <h6 class="mb-1">Environment</h6>
                <p class="mb-0">{{ experiment.env.name }}</p>
                <small class="text-muted">Environment details placeholder</small>
            </div>
        </div>
    </aside>
</template>

<script setup lang="ts">
import { Experiment } from '../../models/Experiment';

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
    width: 300px;
    border: 1px solid var(--bs-border-color);
    border-radius: 0.5rem;
    background: var(--bs-body-bg);
    padding: 0.5rem;
    transition: width 0.2s ease;
    position: relative;
    overflow: hidden;
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

.placeholder-card {
    border: 1px dashed var(--bs-border-color);
    border-radius: 0.5rem;
    padding: 0.75rem;
    background: color-mix(in srgb, var(--bs-body-bg) 94%, var(--bs-secondary-bg) 6%);
}
</style>
