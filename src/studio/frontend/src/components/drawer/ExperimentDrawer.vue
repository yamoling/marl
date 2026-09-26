<script setup lang="ts">
/**
 * Experiment drawer (product-spec §6): header (name, algorithm, created, health, running state
 * with overall progress, Start runs, Stop all, Episodes, ⋯ menu with Rename / Delete / Copy id /
 * Copy folder path) and the Overview, Parameters, Runs and Issues tabs. Works for loaded and
 * not-loaded experiments (opened from the library).
 */
import { computed, ref, watch } from "vue";
import { vTip } from "../../composables/tooltip";
import { useRunActions } from "../../composables/useRunActions";
import { fmtPercent, fmtRelativeDate } from "../../domain/format";
import { useExperimentsStore } from "../../stores/experiments";
import { useReplayStore } from "../../stores/replay";
import { useToasts } from "../../stores/toasts";
import { DRAWER_TABS, useUiStore, type DrawerTab } from "../../stores/ui";
import { copyText } from "../../stores/workspace";
import LaunchButton from "../launch/LaunchButton.vue";
import Drawer from "../shell/Drawer.vue";
import HealthBadge from "../shell/HealthBadge.vue";
import Icon from "../shell/Icon.vue";
import Popover from "../shell/Popover.vue";
import ProgressBar from "../shell/ProgressBar.vue";
import PulseDot from "../shell/PulseDot.vue";
import IssuesTab from "./IssuesTab.vue";
import OverviewTab from "./OverviewTab.vue";
import ParametersTab from "./ParametersTab.vue";
import RunsTab from "./RunsTab.vue";

const ui = useUiStore();
const experiments = useExperimentsStore();
const replay = useReplayStore();
const toasts = useToasts();
const actions = useRunActions();

const id = computed(() => ui.drawerId);
const open = computed({ get: () => !!id.value, set: (v) => !v && ui.closeDrawer() });
const entry = computed(() => (id.value ? experiments.entry(id.value) : undefined));
const detail = computed(() => entry.value?.detail ?? null);
const info = computed(() => (id.value ? experiments.info(id.value) : null));
const loaded = computed(() => !!id.value && experiments.loaded.includes(id.value));
const running = computed(() => info.value?.status === "RUNNING");
const nRunning = computed(() => (id.value ? experiments.runs(id.value).filter((r) => r.status === "RUNNING").length : 0));
const colour = computed(() => (id.value ? experiments.colours[id.value] : undefined));
const menuBtn = ref<HTMLButtonElement | null>(null);
const menuOpen = ref(false);

watch(
    id,
    (v) => {
        if (!v) return;
        if (!experiments.entry(v)) void experiments.fetch(v);
        else experiments.ensureHealth(v);
    },
    { immediate: true },
);
watch(detail, (d) => d && id.value && experiments.ensureHealth(id.value));

const TAB_LABEL: Record<DrawerTab, string> = { overview: "Overview", params: "Parameters", runs: "Runs", issues: "Issues" };
const tabCount = (t: DrawerTab) => (t === "runs" ? detail.value?.runs.length : t === "issues" ? detail.value?.issues.length : undefined);
const hasErrors = computed(() => !!detail.value?.issues.some((i) => i.level === "error"));
const busyTip = "Stop the running runs first";

function menu(action: () => void): void {
    menuOpen.value = false;
    action();
}
async function copy(text: string, what: string): Promise<void> {
    await copyText(text);
    toasts.push({ message: `Copied ${what}` });
}
const folder = computed(() => {
    const l = detail.value?.raw.logdir;
    return typeof l === "string" && l ? l : `logs/${id.value}`;
});
/** Open the episodes sheet at the last test step (the replay store resolves `null`). */
function openEpisodes(): void {
    if (!id.value) return;
    replay.openEpisodes(id.value, null);
}
</script>

<template>
    <Drawer v-model:open="open" wide :title="id ?? ''">
        <template #header>
            <div v-if="id" class="dh">
                <div class="eyebrow">
                    <span class="dot" :style="{ '--c': colour ?? 'var(--ink3)' }" />Experiment
                    <span v-if="!loaded" class="tag">not loaded</span>
                </div>
                <h2 :title="id">{{ detail?.name ?? id.split("/").pop() }}</h2>
                <div v-if="id.includes('/')" class="mono muted small">{{ id }}</div>
                <div class="meta">
                    <span v-if="detail?.algo" class="algo">{{ detail.algo }}</span>
                    <span v-if="detail" class="muted">created {{ fmtRelativeDate(detail.created) }}</span>
                    <HealthBadge v-if="detail" :health="detail.health" :counts="detail.issue_counts" :tip="experiments.healthText(id)" />
                    <span v-if="entry?.status === 'missing'" class="hb error">✖ not found</span>
                    <span v-if="running" class="hb run live"
                        ><PulseDot :size="6" /> {{ nRunning }} running · {{ fmtPercent(info?.progress) }}</span
                    >
                </div>
                <ProgressBar v-if="running" class="prog" :value="info?.progress ?? null" show-label />
                <div v-if="detail" class="acts">
                    <LaunchButton :id="id" />
                    <button v-if="running" type="button" class="btn" data-act="stop-all" @click="actions.stopAll(id, nRunning)">
                        <Icon name="stop" :size="12" />Stop all runs
                    </button>
                    <button type="button" class="btn" v-tip="'Episodes and replay at the last test step'" @click="openEpisodes">
                        <Icon name="film" :size="13" />Episodes
                    </button>
                    <button v-if="!loaded" type="button" class="btn soft" @click="experiments.load(id)">
                        <Icon name="plus" :size="13" />Load
                    </button>
                    <button ref="menuBtn" type="button" class="btn" aria-label="More actions" data-act="more" @click="menuOpen = !menuOpen">
                        <Icon name="more" />
                    </button>
                </div>
            </div>
        </template>

        <template v-if="id">
            <div v-if="!detail" class="empty">
                <template v-if="entry?.status === 'missing'">This experiment no longer exists (deleted or renamed).</template>
                <template v-else-if="entry?.status === 'error'">Could not load: {{ entry.error }}</template>
                <template v-else>Loading…</template>
            </div>
            <template v-else>
                <nav class="tabs" role="tablist">
                    <button
                        v-for="t in DRAWER_TABS"
                        :key="t"
                        type="button"
                        role="tab"
                        :aria-selected="ui.drawerTab === t"
                        :class="{ on: ui.drawerTab === t }"
                        :data-tab="t"
                        @click="ui.drawerTab = t"
                    >
                        {{ TAB_LABEL[t]
                        }}<span v-if="tabCount(t)" class="n" :class="{ error: t === 'issues' && hasErrors }">{{ tabCount(t) }}</span>
                    </button>
                </nav>
                <div class="tab-body">
                    <OverviewTab v-if="ui.drawerTab === 'overview'" :detail="detail" />
                    <ParametersTab v-else-if="ui.drawerTab === 'params'" :key="detail.id" :detail="detail" />
                    <RunsTab v-else-if="ui.drawerTab === 'runs'" :detail="detail" />
                    <IssuesTab v-else :detail="detail" />
                </div>
            </template>
        </template>

        <Popover v-model:open="menuOpen" :anchor="menuBtn" placement="bottom-end" :min-width="220" label="Experiment actions">
            <template v-if="id">
                <span v-tip="running ? busyTip : ''"
                    ><button type="button" class="mi" :disabled="running" data-act="rename" @click="menu(() => actions.rename(id!))">
                        <Icon name="pencil" :size="13" />Rename…
                    </button></span
                >
                <span v-tip="running ? busyTip : ''"
                    ><button type="button" class="mi danger" :disabled="running" data-act="delete" @click="menu(() => actions.remove(id!))">
                        <Icon name="trash" :size="13" />Delete…
                    </button></span
                >
                <hr />
                <button type="button" class="mi" @click="menu(() => copy(id!, 'the experiment id'))">
                    <Icon name="copy" :size="13" />Copy id
                </button>
                <button type="button" class="mi" @click="menu(() => copy(folder, 'the folder path'))">
                    <Icon name="copy" :size="13" />Copy folder path<small class="mono">{{ folder }}</small>
                </button>
                <template v-if="loaded">
                    <hr />
                    <button type="button" class="mi" @click="menu(() => experiments.unload(id!))">
                        <Icon name="x" :size="13" />Unload from workspace
                    </button>
                </template>
            </template>
        </Popover>
    </Drawer>
</template>

<style scoped>
.dh h2 {
    margin: 2px 0 2px;
    font-size: 20px;
    letter-spacing: -0.02em;
    word-break: break-all;
}
.small {
    font-size: 11.5px;
}
.meta {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 4px;
    font-size: 12px;
}
.algo {
    font-size: 10.5px;
    font-weight: 700;
    padding: 1px 7px;
    border-radius: 6px;
    background: #f0f0f5;
    color: var(--ink2);
}
.tag {
    font-size: 10px;
    color: var(--acc2);
    background: var(--acc-soft);
    border-radius: 5px;
    padding: 0 6px;
    text-transform: none;
    letter-spacing: 0;
}
.live {
    display: inline-flex;
    align-items: center;
    gap: 5px;
}
.prog {
    margin-top: 6px;
    max-width: 360px;
}
.acts {
    display: flex;
    gap: 8px;
    margin-top: 12px;
    flex-wrap: wrap;
}
.tabs {
    display: flex;
    gap: 2px;
    border-bottom: 1px solid var(--line);
    margin: 0 -22px 14px;
    padding: 0 22px;
    position: sticky;
    top: -4px;
    background: var(--card);
    z-index: 2;
}
.tabs button {
    border: 0;
    background: none;
    padding: 9px 12px;
    color: var(--ink2);
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    font-weight: 550;
}
.tabs button.on {
    color: var(--acc2);
    border-bottom-color: var(--acc);
}
.tabs .n {
    margin-left: 6px;
    font-size: 10.5px;
    background: #f0f0f5;
    border-radius: 999px;
    padding: 0 6px;
    color: var(--ink2);
}
.tabs .n.error {
    background: var(--err-soft);
    color: var(--err);
}
.mi.danger:hover:not(:disabled) {
    background: var(--err-soft);
    color: var(--err);
}
.mi small.mono {
    max-width: 140px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
</style>
