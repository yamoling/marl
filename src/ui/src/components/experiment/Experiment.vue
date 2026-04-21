<template>
    <div class="experiment-panel">
        <div v-if="loadError" class="experiment-load-error">
            <font-awesome-icon :icon="['fas', 'exclamation-triangle']" class="error-icon" />
            <h3>Failed to load experiment</h3>
            <p class="text-muted">{{ loadError }}</p>
            <button class="btn btn-outline-primary btn-sm" @click="() => router.push('/home')">
                Back to home
            </button>
        </div>
        <template v-else>
            <ExperimentDetailsPane v-if="experiment != null" :experiment="experiment" :is-open="isDetailsPaneOpen"
                @toggle="toggleDetailsPane" />
            <div class="workspace" :class="{ 'with-replay': showReplayPane }">
                <section class="workspace-main">
                    <MetricsTable :logdir="logdir" @view-episode="onViewEpisode" />
                </section>

                <section v-show="showReplayPane" class="workspace-replay">
                    <div class="inline-replay">
                        <EpisodeReplay ref="episodeReplay" :logdir="logdir" :experiment="experiment"
                            @close="() => (showReplayPane = false)" />
                    </div>
                </section>
            </div>
        </template>
    </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
import { Experiment } from "../../models/Experiment";
import MetricsTable from "./MetricsTable.vue";
import { useRoute, useRouter } from "vue-router";
import { useExperimentStore } from "../../stores/ExperimentStore";
import EpisodeReplay from "../visualisation/EpisodeReplay.vue";
import ExperimentDetailsPane from "./ExperimentDetailsPane.vue";

const route = useRoute();
const router = useRouter();
const logdir = (route.params.logdir as string[]).join("/");
const experiment = ref(null as Experiment | null);
const experimentStore = useExperimentStore();
const isDetailsPaneOpen = ref(false);
const loading = ref(true);
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

function onViewEpisode(episodeDirectory: string) {
    showReplayPane.value = true;
    episodeReplay.value.load(episodeDirectory);
}

onMounted(async () => {
    loading.value = true;
    window.addEventListener("keydown", onEscapePressed);
    const res = await experimentStore.getExperiment(logdir);
    if (res == null) {
        loadError.value = `Could not load experiment at "${logdir}". Check the error notification for details.`;
        loading.value = false;
        return;
    }
    experiment.value = res;
    loading.value = false;
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

.experiment-load-error {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    min-height: 50vh;
    text-align: center;
}

.experiment-load-error .error-icon {
    font-size: 3rem;
    color: var(--bs-danger);
}

@media (max-width: 1200px) {
    .workspace.with-replay {
        grid-template-columns: minmax(0, 1fr);
    }
}
</style>
