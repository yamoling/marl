<script setup lang="ts">
/**
 * MARL Studio main page: top bar, fields panel (left) and the plot workspace. Restores the
 * selected workspace's plots, fetches loaded experiments, connects live events and system readings.
 * Keyboard: Ctrl/⌘+K opens the library and focuses its search.
 */
import { onBeforeUnmount, onMounted, ref } from "vue";

import FieldsPanel from "../components/fields/FieldsPanel.vue";
import LibrarySheet from "../components/library/LibrarySheet.vue";
import PlotGrid from "../components/plots/PlotGrid.vue";
import PresetsPanel from "../components/plots/PresetsPanel.vue";
import SettingsDialog from "../components/settings/SettingsDialog.vue";
import Icon from "../components/shell/Icon.vue";
import TopBar from "../components/shell/TopBar.vue";
import ParamDiffOverlay from "../components/diff/ParamDiffOverlay.vue";
import ExperimentDrawer from "../components/drawer/ExperimentDrawer.vue";
import RenameDialog from "../components/launch/RenameDialog.vue";
import StartRunsDialog from "../components/launch/StartRunsDialog.vue";
import ConfirmHost from "../components/shell/ConfirmHost.vue";
import { usePlotActions } from "../composables/usePlotActions";
import { useUrlState } from "../composables/useUrlState";
import { useExperimentsStore } from "../stores/experiments";
import { useLibraryStore } from "../stores/library";
import { useLiveStore } from "../stores/live";
import { useSystemStore } from "../stores/system";
import { useWorkspaceStore } from "../stores/workspace";

const workspace = useWorkspaceStore();

const experiments = useExperimentsStore();
const library = useLibraryStore();
const live = useLiveStore();
const system = useSystemStore();
const actions = usePlotActions();
const settingsOpen = ref(false);
const librarySheet = ref<InstanceType<typeof LibrarySheet> | null>(null);

experiments.ensureAll();
useUrlState();

function onKey(ev: KeyboardEvent): void {
    if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "k") {
        ev.preventDefault();
        if (library.open) librarySheet.value?.focusSearch();
        else library.show();
    }
}

onMounted(() => {
    live.connect();
    system.start();
    window.addEventListener("keydown", onKey);
    window.addEventListener("beforeunload", workspace.persistNow);
});
onBeforeUnmount(() => {
    workspace.persistNow();
    live.disconnect();
    system.stop();
    window.removeEventListener("keydown", onKey);
    window.removeEventListener("beforeunload", workspace.persistNow);
});
</script>

<template>
    <div class="studio">
        <TopBar @settings="settingsOpen = true" />
        <div class="main">
            <FieldsPanel />
            <section class="ws" aria-label="Workspace">
                <div class="ws-head">
                    <span class="sub"
                        >{{ workspace.plots.length }} plot{{ workspace.plots.length === 1 ? "" : "s" }} ·
                        {{ experiments.loaded.length }} experiment{{ experiments.loaded.length === 1 ? "" : "s" }}</span
                    >
                    <span class="sp" />
                    <span v-if="experiments.loaded.length" class="muted hint">Drag fields from the left onto a plot, or click them</span>
                    <button type="button" class="btn" @click="actions.newEmptyPlot()"><Icon name="plus" :size="14" />New plot</button>
                </div>

                <div v-if="!experiments.loaded.length && !workspace.plots.length" class="welcome">
                    <div class="logo"><Icon name="chart" :size="26" /></div>
                    <h2>Welcome to MARL Studio</h2>
                    <p>Load one or more experiments to explore their metrics, compare parameters and build plots.</p>
                    <button type="button" class="btn primary" @click="library.show()">
                        <Icon name="plus" :size="14" />Add experiments
                    </button>
                    <p class="muted small">Tip: press <kbd>Ctrl</kbd> + <kbd>K</kbd> to search the library.</p>
                </div>
                <template v-else>
                    <PresetsPanel v-if="!workspace.plots.length && experiments.loaded.length" />
                    <PlotGrid />
                </template>
            </section>
        </div>
        <LibrarySheet ref="librarySheet" />
        <SettingsDialog v-model:open="settingsOpen" />
        <ExperimentDrawer />
        <ParamDiffOverlay />
        <StartRunsDialog />
        <RenameDialog />
        <ConfirmHost />
    </div>
</template>

<style scoped>
.studio {
    height: 100%;
    display: grid;
    grid-template-rows: auto minmax(0, 1fr);
    overflow: hidden;
}
.main {
    display: grid;
    grid-template-columns: 272px minmax(0, 1fr);
    min-height: 0;
}
.ws {
    overflow: auto;
    padding: 20px 26px 90px;
}
.ws-head {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
}

.sub {
    color: var(--ink3);
}
.sp {
    flex: 1;
}
.hint {
    font-size: 12px;
}
.welcome {
    max-width: 480px;
    margin: 60px auto;
    text-align: center;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: var(--r);
    box-shadow: var(--sh);
    padding: 32px 28px;
}
.welcome .logo {
    width: 48px;
    height: 48px;
    margin: 0 auto 10px;
    border-radius: 14px;
    background: var(--logo-grad);
    color: #fff;
    display: grid;
    place-items: center;
}
.welcome h2 {
    margin: 0 0 6px;
}
.welcome p {
    color: var(--ink2);
}
.small {
    font-size: 12px;
}
kbd {
    font-family: var(--mono);
    font-size: 11px;
    border: 1px solid var(--line2);
    border-bottom-width: 2px;
    border-radius: 4px;
    padding: 0 4px;
    background: #fafafc;
}
</style>
