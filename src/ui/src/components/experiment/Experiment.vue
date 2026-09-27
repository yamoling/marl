<template>
    <div class="experiment-panel">
        <ExperimentDetailsPane
            v-if="experiment != null"
            :experiment="experiment"
            :is-open="isDetailsPaneOpen"
            @toggle="toggleDetailsPane"
        />
        <div class="workspace" :class="{ 'with-replay': showReplayPane }">
            <section class="workspace-main">
                <div v-if="loadError" class="alert alert-warning mb-2" role="alert">
                    Experiment metadata could not be loaded for "{{ logdir }}" ({{ loadError }}). Results may still be available below;
                    check the experiment metadata and reload the page to retry.
                </div>
                <MetricsTable :logdir="logdir" @view-episode="onViewEpisode" />
            </section>

            <section v-show="showReplayPane" class="workspace-replay">
                <div class="inline-replay">
                    <EpisodeReplay ref="episodeReplay" :logdir="logdir" :experiment="experiment" @close="() => (showReplayPane = false)" />
                </div>
            </section>
        </div>
    </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
import { Experiment } from "../../models/Experiment";
import MetricsTable from "./MetricsTable.vue";
import { useRoute } from "vue-router";
import { useExperimentStore } from "../../stores/ExperimentStore";
import EpisodeReplay from "../visualisation/EpisodeReplay.vue";
import ExperimentDetailsPane from "./ExperimentDetailsPane.vue";
import { ReplayEpisodeSummary } from "../../models/Episode";

const route = useRoute();

const logdir = (route.params.logdir as string[]).join("/");
const experiment = ref(null as Experiment | null);
const experimentStore = useExperimentStore();
const isDetailsPaneOpen = ref(false);

const loadError = ref<string | null>(null);
const showReplayPane = ref(false);
const episodeReplay = ref();

function toggleDetailsPane() {
    isDetailsPaneOpen.value = !isDetailsPaneOpen.value;
}

function onEscapePressed(event: KeyboardEvent) {
    if (event.key === "Escape") {
        showReplayPane.value = false;
    }
}

function onViewEpisode(summary: ReplayEpisodeSummary) {
    showReplayPane.value = true;
    episodeReplay.value.load(summary);
}

/** Load metadata independently so unavailable metadata does not hide existing results. @ai-edited */
onMounted(async () => {
    window.addEventListener("keydown", onEscapePressed);
    try {
        experiment.value = await experimentStore.getExperiment(logdir);
        if (experiment.value == null) {
            loadError.value = "The server returned no experiment metadata.";
        }
    } catch (e) {
        loadError.value = e instanceof Error ? e.message : String(e);
    }
});

onUnmounted(() => window.removeEventListener("keydown", onEscapePressed));
</script>

<style scoped>
.experiment-panel {
    display: flex;
    gap: 0.75rem;
    min-height: 76vh;
}

.workspace {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    flex: 1;
    gap: 0.75rem;
    min-width: 0;
}

.workspace.with-replay {
    grid-template-columns: minmax(20vw, 1fr) minmax(0, 4fr);
}

.workspace-main,
.workspace-replay {
    background: var(--bs-body-bg);
    border: 1px solid var(--bs-border-color);
    border-radius: 0.5rem;
    padding: 0.75rem;
}

.workspace-main {
    min-width: 20vw;
    overflow-y: auto;
    overflow-x: hidden;
}

.workspace-replay {
    width: 100%;
    overflow: auto;
    min-width: 0;
}

.inline-replay {
    height: 100%;
    overflow: auto;
}

.table-scroll,
.plot-scroll {
    max-height: 72vh;
    min-width: 0;
    overflow-y: auto;
    overflow-x: auto;
}

@media (max-width: 1200px) {
    .workspace.with-replay {
        grid-template-columns: minmax(0, 1fr);
    }
}
</style>
