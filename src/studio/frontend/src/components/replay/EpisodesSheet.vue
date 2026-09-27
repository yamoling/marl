<script setup lang="ts">
/**
 * Episodes & replay bottom sheet (product spec §8): experiment selector, performance timeline
 * with its metric selector and step arrows, episode cards grouped by seed, replay viewer.
 * Its state is mirrored in the URL query (`?episodes=<id>@<step>`); `Esc` closes it.
 */
import { computed, onMounted, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { groupBySeed, scoreKey } from "../../domain/replay";
import { formatEpisodesParam, parseEpisodesParam, timelineMetricOptions } from "../../domain/timeline";
import { useExperimentsStore } from "../../stores/experiments";
import { useReplayStore } from "../../stores/replay";
import Icon from "../shell/Icon.vue";
import Sheet from "../shell/Sheet.vue";
import EpisodeCard from "./EpisodeCard.vue";
import PerformanceTimeline from "./PerformanceTimeline.vue";
import ReplayViewer from "./ReplayViewer.vue";

const store = useReplayStore();
const experiments = useExperimentsStore();
const route = useRoute();
const router = useRouter();

const exp = computed(() => store.experiment);
const choices = computed(() => {
    const ids = [...experiments.loaded];
    if (exp.value && !ids.includes(exp.value)) ids.unshift(exp.value);
    return ids;
});
const colour = computed(() => (exp.value ? (experiments.colours[exp.value] ?? "var(--acc)") : "var(--acc)"));
const options = computed(() => timelineMetricOptions(store.catalog));
const groups = computed(() => groupBySeed(store.episodes));
const score = computed(() => scoreKey(store.episodes, store.metric?.metric));
const stepIndex = computed(() => (store.step === null ? -1 : store.steps.indexOf(store.step)));
/** Runs of the experiment without episodes at this step (not reached yet, or no test table). */
const missingRuns = computed(() => {
    if (store.episodesStatus !== "ok" || !exp.value) return [];
    const have = new Set(store.episodes.map((e) => e.run));
    return experiments.runs(exp.value).filter((r) => !have.has(r.id));
});
const shortName = (id: string) => experiments.names[id] ?? id.split("/").pop() ?? id;

function onStepInput(ev: Event): void {
    const v = Number((ev.target as HTMLInputElement).value);
    if (Number.isFinite(v)) store.setStep(v);
    (ev.target as HTMLInputElement).value = String(store.step ?? "");
}

// ---------------------------------------------------------------- URL sync

/** Navigations started by `writeQuery` still in flight: their route changes are not external. */
let writing = 0;

/** Write the sheet state into `?episodes=` (replace, other query keys kept). @ai-generated */
function writeQuery(): void {
    const want = store.open && store.experiment ? formatEpisodesParam(store.experiment, store.step) : undefined;
    if ((route.query.episodes ?? undefined) === want) return;
    const query = { ...route.query };
    if (want === undefined) delete query.episodes;
    else query.episodes = want;
    writing++;
    void router.replace({ query }).finally(() => writing--);
}

/** Apply `?episodes=` when it differs from the state (page load, back/forward, pasted link). @ai-generated */
function readQuery(): void {
    if (writing) return;
    const p = parseEpisodesParam(route.query.episodes);
    if (!p) {
        if (store.open) store.close();
        return;
    }
    if (store.open && p.experiment === store.experiment && (p.step === null || p.step === store.step)) return;
    store.openEpisodes(p.experiment, p.step);
}

onMounted(() => {
    void router.isReady().then(() => {
        readQuery();
        watch(() => [store.open, store.experiment, store.step], writeQuery);
        watch(() => route.query.episodes, readQuery);
    });
});
</script>

<template>
    <Sheet :open="store.open" variant="bottom" title="Episodes" @update:open="(v) => !v && store.close()">
        <template #header>
            <div class="head">
                <div class="titles">
                    <div class="eyebrow">Episodes &amp; replay</div>
                    <h2>
                        Episodes at step <span class="mono">{{ store.step === null ? "—" : store.step.toLocaleString("en-US") }}</span>
                    </h2>
                </div>
                <label class="xsel">
                    <span class="dot" :style="{ '--c': colour }" />
                    <select
                        :value="exp ?? ''"
                        aria-label="Experiment"
                        @change="store.setExperiment(($event.target as HTMLSelectElement).value)"
                    >
                        <option v-for="id in choices" :key="id" :value="id">
                            {{ shortName(id) }}{{ experiments.loaded.includes(id) ? "" : " (not loaded)" }}
                        </option>
                    </select>
                </label>
                <div class="stepper" role="group" aria-label="Test step">
                    <button type="button" aria-label="Previous test step" :disabled="stepIndex <= 0" @click="store.stepBy(-1)">
                        <Icon name="chevron-left" :size="15" />
                    </button>
                    <input
                        :value="store.step ?? ''"
                        inputmode="numeric"
                        aria-label="Step"
                        @change="onStepInput"
                        @keydown.enter="onStepInput"
                    />
                    <button
                        type="button"
                        aria-label="Next test step"
                        :disabled="stepIndex < 0 || stepIndex >= store.steps.length - 1"
                        @click="store.stepBy(1)"
                    >
                        <Icon name="chevron-right" :size="15" />
                    </button>
                </div>
                <span class="muted small">{{
                    store.steps.length
                        ? `${stepIndex + 1} / ${store.steps.length} test steps`
                        : store.stepsStatus === "loading"
                          ? "loading steps…"
                          : "no test steps"
                }}</span>
            </div>
        </template>

        <div v-if="exp" class="episodes-body">
            <PerformanceTimeline
                :experiment="exp"
                :steps="store.steps"
                :step="store.step"
                :metric="store.metric"
                :options="options"
                :colour="colour"
                @step="store.setStep"
                @nudge="store.stepBy"
                @metric="store.setMetric"
            />
            <div class="grid">
                <div class="list" aria-label="Episodes">
                    <p v-if="store.episodesStatus === 'loading' && !store.episodes.length" class="muted">Loading episodes…</p>
                    <p v-else-if="store.episodesStatus === 'error'" class="note error">
                        Could not load the episodes: {{ store.episodesError }}
                    </p>
                    <div v-else-if="store.episodesStatus === 'ok' && !store.episodes.length" class="empty">
                        No test episode at this step{{ store.steps.length ? "" : " (the experiment has no test data)" }}.
                    </div>
                    <section v-for="g in groups" :key="g.key" class="seed" :class="{ stale: store.episodesStatus === 'loading' }">
                        <h4>
                            seed {{ g.seed ?? "?" }} <span class="muted">{{ g.run.split("/").pop() }}</span>
                        </h4>
                        <div class="cards">
                            <EpisodeCard
                                v-for="e in g.episodes"
                                :key="`${e.run}|${e.test}`"
                                :episode="e"
                                :score-key="score"
                                :selected="store.isSelected(e)"
                                @select="store.selectEpisode(e)"
                            />
                        </div>
                    </section>
                    <p v-if="missingRuns.length && store.episodes.length" class="note warn small">
                        ⚠ No test episode at this step for
                        {{ missingRuns.map((r) => (r.seed === null ? r.dirname : `seed ${r.seed}`)).join(", ") }}.
                    </p>
                    <p v-if="store.badEpisodes" class="note warn small">
                        ⚠ {{ store.badEpisodes }} episode{{ store.badEpisodes > 1 ? "s" : "" }} could not be read.
                    </p>
                </div>
                <div class="view">
                    <ReplayViewer />
                </div>
            </div>
        </div>
    </Sheet>
</template>

<style scoped>
.head {
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
}
.titles {
    margin-right: auto;
}
.titles h2 {
    margin: 2px 0 0;
    font-size: 19px;
    letter-spacing: -0.02em;
}
.xsel {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    border: 1px solid var(--line2);
    border-radius: 999px;
    padding: 2px 4px 2px 10px;
    background: var(--card);
}
.xsel select {
    border: 0;
    background: transparent;
    font-weight: 600;
    max-width: 320px;
}
.stepper {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    background: var(--seg-bg);
    border-radius: 9px;
    padding: 2px;
}
.stepper button {
    border: 0;
    background: none;
    width: 26px;
    height: 26px;
    border-radius: 7px;
    display: grid;
    place-items: center;
}
.stepper button:hover:not(:disabled) {
    background: var(--card);
}
.stepper button:disabled {
    opacity: 0.35;
}
.stepper input {
    width: 92px;
    border: 0;
    background: var(--card);
    border-radius: 7px;
    padding: 3px 8px;
    text-align: right;
    font-family: var(--mono);
}
.small {
    font-size: var(--fs-xs);
}
.episodes-body {
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-height: 0;
}
.grid {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: minmax(280px, 400px) minmax(0, 1fr);
    gap: 16px;
}
.list,
.view {
    overflow: auto;
    min-height: 0;
}
.list {
    padding-right: 4px;
}
.view {
    border-left: 1px solid var(--line);
    padding-left: 16px;
}
.seed {
    margin-bottom: 12px;
    transition: opacity 0.15s;
}
.seed.stale {
    opacity: 0.5;
}
.seed h4 {
    margin: 0 0 6px;
    font-size: 12.5px;
    font-weight: 600;
    color: var(--ink2);
    display: flex;
    gap: 8px;
}
.cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 6px;
}
.empty {
    padding: 30px 10px;
}
</style>

<style>
/* The episodes sheet needs more room than the default bottom sheet. */
section.sheet.bottom:has(.episodes-body) {
    height: min(920px, 92vh);
}
</style>
