<script setup lang="ts">
/**
 * Global running indicator "● N runs running ▾" with a popover listing running experiments
 * (loaded or not) and their runs, with Stop actions (confirmed) per run and per experiment.
 */
import { computed, ref } from "vue";
import { vTip } from "../../composables/tooltip";
import { useRunActions } from "../../composables/useRunActions";
import { useUiStore } from "../../stores/ui";
import { fmtPercent, fmtStep, shortName } from "../../domain/format";
import { useExperimentsStore } from "../../stores/experiments";
import { useLiveStore } from "../../stores/live";
import Icon from "./Icon.vue";
import Popover from "./Popover.vue";
import ProgressBar from "./ProgressBar.vue";
import PulseDot from "./PulseDot.vue";

const live = useLiveStore();
const experiments = useExperimentsStore();
const open = ref(false);
const btn = ref<HTMLButtonElement | null>(null);
const n = computed(() => live.running.length);
const groups = computed(() =>
    Object.entries(live.runningByExperiment).map(([id, runs]) => ({
        id,
        name: experiments.loaded.includes(id) ? experiments.name(id) : shortName(id),
        loaded: experiments.loaded.includes(id),
        colour: experiments.colours[id],
        // Same definition as the pills (mean over all runs) when the experiment is loaded.
        progress: experiments.info(id)?.progress ?? runs.reduce((a, r) => a + (r.progress ?? 0), 0) / runs.length,
        runs,
    })),
);
const actions = useRunActions();
const ui = useUiStore();
function details(id: string): void {
    open.value = false;
    ui.openDrawer(id, "runs");
}
</script>

<template>
    <button ref="btn" class="running-badge" :class="{ idle: !n }" :aria-expanded="open" @click="open = !open">
        <template v-if="n"><PulseDot /> {{ n }} run{{ n > 1 ? "s" : "" }} running</template>
        <template v-else><span class="idle-dot" /> idle</template>
        <Icon name="chevron-down" :size="13" />
    </button>
    <Popover v-model:open="open" :anchor="btn" placement="bottom-end" :min-width="320" label="Running experiments">
        <h4>Running</h4>
        <div v-if="!groups.length" class="none">Nothing is running.</div>
        <div v-for="g in groups" :key="g.id" class="grp">
            <div class="gh">
                <span class="dot" :style="{ '--c': g.colour ?? 'var(--ink3)' }" />
                <button type="button" class="nm" v-tip="`${g.id} — open details`" @click="details(g.id)">{{ g.name }}</button>
                <span v-if="!g.loaded" class="muted">not loaded</span>
                <span class="sp" />
                <button class="btn small" data-act="stop-all" @click="actions.stopAll(g.id, g.runs.length)">Stop all</button>
            </div>
            <ProgressBar :value="g.progress" show-label />
            <div v-for="r in g.runs" :key="r.run" class="run">
                <span class="rn">{{ r.run.split("/").pop() }}</span>
                <ProgressBar :value="r.progress" :height="4" class="rp" />
                <span class="muted st">{{ fmtPercent(r.progress) }} · {{ fmtStep(r.latest_step) }}</span>
                <button class="hbtn danger" v-tip="'Stop this run'" :aria-label="`Stop ${r.run}`" @click="actions.stopRun(g.id, r.run)">
                    <Icon name="stop" :size="13" />
                </button>
            </div>
        </div>
    </Popover>
</template>

<style scoped>
.nm {
    border: 0;
    background: none;
    padding: 0;
    font-weight: 700;
}
.nm:hover {
    color: var(--acc2);
    text-decoration: underline;
}
.running-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    border: 1px solid var(--acc-soft2);
    background: var(--acc-soft);
    color: var(--acc2);
    border-radius: 999px;
    padding: 4px 10px;
    font-weight: 600;
    font-size: var(--fs-sm);
    white-space: nowrap;
}
.running-badge.idle {
    background: var(--card);
    border-color: var(--line2);
    color: var(--ink3);
    font-weight: 500;
}
.idle-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--line2);
}
.none {
    padding: 4px 10px 8px;
    color: var(--ink3);
}
.grp {
    padding: 6px 10px 8px;
    border-top: 1px solid var(--line);
}
.gh {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
}
.sp {
    flex: 1;
}
.run {
    display: grid;
    grid-template-columns: 56px 1fr auto auto;
    gap: 8px;
    align-items: center;
    font-size: 11.5px;
    margin-top: 3px;
}
.rn {
    font-family: var(--mono);
    color: var(--ink2);
}
.st {
    font-size: 10.5px;
    white-space: nowrap;
}
</style>
