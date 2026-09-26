<script setup lang="ts">
/** Top bar: brand, experiment pills, add/compare actions, running indicator, live status, settings. */
import { computed } from "vue";
import { vTip } from "../../composables/tooltip";
import { useExperimentsStore } from "../../stores/experiments";
import { useLibraryStore } from "../../stores/library";
import { useLiveStore } from "../../stores/live";
import { useUiStore } from "../../stores/ui";
import ExperimentPill from "./ExperimentPill.vue";
import Icon from "./Icon.vue";
import RunningBadge from "./RunningBadge.vue";

const emit = defineEmits<{ settings: [] }>();
const experiments = useExperimentsStore();
const library = useLibraryStore();
const live = useLiveStore();
const ui = useUiStore();

const canCompare = computed(() => experiments.loaded.length >= 2);
const compareTip = computed(() =>
    canCompare.value ? "Compare the parameters of the loaded experiments" : "Load at least 2 experiments to compare their parameters",
);

function openDrawer(id: string): void {
    ui.openDrawer(id);
}
</script>

<template>
    <header class="top">
        <div class="brand">
            <span class="logo"><Icon name="logo" :size="13" /></span>MARL Studio
        </div>
        <div class="xpills">
            <ExperimentPill v-for="id in experiments.loaded" :id="id" :key="id" @open="openDrawer" />
            <span v-if="!experiments.loaded.length" class="muted">No experiments loaded</span>
        </div>
        <button class="btn soft" v-tip="'Open the library (Ctrl+K)'" @click="library.show()">
            <Icon name="plus" :size="14" />Add experiments
        </button>
        <span v-tip="compareTip"
            ><button class="btn" :disabled="!canCompare" data-act="compare" @click="ui.diffOpen = true">
                <Icon name="compare" :size="14" />Compare parameters
            </button></span
        >
        <RunningBadge />
        <span
            v-if="live.paused"
            class="paused"
            v-tip="'Live updates paused — reconnecting…'"
            role="status"
            aria-label="Live updates paused"
        />
        <button class="hbtn" v-tip="'Settings'" aria-label="Settings" @click="emit('settings')"><Icon name="settings" /></button>
    </header>
</template>

<style scoped>
.top {
    position: relative;
    z-index: var(--z-top);
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 18px;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--line);
}
.brand {
    font-weight: 700;
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 15px;
    white-space: nowrap;
}
.logo {
    width: 26px;
    height: 26px;
    border-radius: 8px;
    background: var(--logo-grad);
    display: grid;
    place-items: center;
    color: #fff;
    box-shadow: 0 3px 10px rgba(86, 70, 224, 0.35);
}
.xpills {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    align-items: center;
    flex: 1;
    min-width: 0;
}
.paused {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--warn);
    box-shadow: 0 0 0 3px var(--warn-soft);
}
</style>
