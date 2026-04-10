<template>
    <div>
        <RouterLink class="btn btn-sm btn-success me-1 mb-1" :to="`/inspect/${logdir}`" @click.stop
            title="Inspect experiment">
            <font-awesome-icon :icon="['fas', 'arrow-up-right-from-square']" />
        </RouterLink>
        <button v-if="hasResults" class="btn btn-sm btn-outline-primary me-1 mb-1" @click.stop="emit('download')"
            title="Download datasets">
            <font-awesome-icon :icon="['fas', 'download']" />
        </button>
        <button v-if="isLoaded" class="btn btn-sm btn-danger me-1 mb-1" @click.stop="emit('unload')"
            title="Unload results">
            <font-awesome-icon :icon="['far', 'circle-xmark']" />
        </button>
        <input v-if="isLoaded" type="color" class="form-control form-control-color d-inline-block align-middle mb-1"
            style="width: 2.5rem; min-width: 2.5rem;" :value="colour" @click.stop @input="onColourInput"
            title="Change experiment colour" aria-label="Change experiment colour" />
    </div>
</template>

<script setup lang="ts">
import { RouterLink } from 'vue-router';

const props = defineProps<{
    logdir: string
    isLoaded: boolean
    hasResults: boolean
    colour: string
}>();

const emit = defineEmits<{
    (event: 'download'): void
    (event: 'unload'): void
    (event: 'change-colour', colour: string): void
}>();

function onColourInput(event: Event) {
    const target = event.target as HTMLInputElement | null;
    if (target == null || target.value.length === 0) {
        return;
    }
    emit('change-colour', target.value);
}
</script>
