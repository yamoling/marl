<script setup lang="ts">
/**
 * Replay of the selected episode (port of the old `EpisodeReplay.vue`): frame, play/pause, scrub
 * slider and step counter, action panel, timeline tracks (+ track wizard), agent-wise information
 * and the "replay mismatch" callout. Keyboard when focused: Space play/pause, ←/→ step,
 * Home/End. When the experiment cannot be replayed, a callout quotes the blocking issue.
 */
import { computed, ref } from "vue";
import { vTip } from "../../composables/tooltip";
import { computeTracks, episodeLength, formatNumber, frameSrc, nAgents } from "../../domain/replay";
import { findTrack, type Track } from "../../domain/timeline";
import type { TrackKind } from "../../domain/settings";
import { PLAYBACK_SPEEDS, useReplayStore } from "../../stores/replay";
import { useTracksStore } from "../../stores/tracks";
import Icon from "../shell/Icon.vue";
import ActionPanel from "./ActionPanel.vue";
import AgentInfo from "./AgentInfo.vue";
import TimelineTrack from "./TimelineTrack.vue";
import ReplayUnavailable from "./ReplayUnavailable.vue";
import TrackWizard from "./TrackWizard.vue";

const store = useReplayStore();
const tracksStore = useTracksStore();
const wizardOpen = ref(false);

const replay = computed(() => store.replay);
const episode = computed(() => {
    const s = store.selected;
    return s ? (store.episodes.find((e) => e.run === s.run && e.test === s.test && e.step === s.step) ?? null) : null;
});
const exp = computed(() => store.experiment ?? "");
/** Replay is known to be impossible, or the backend refused it (409 with the blocking issue). */
const gated = computed(() => store.canReplay === false || (store.replayStatus === "error" && store.replayError?.status === 409));
const agents = computed(() => (replay.value ? nAgents(replay.value) : 0));
const length = computed(() => (replay.value ? episodeLength(replay.value) : 0));
const frame = computed(() => (replay.value ? frameSrc(replay.value.frames[Math.min(store.t, replay.value.frames.length - 1)]) : ""));

const allTracks = computed(() => (replay.value ? computeTracks(replay.value) : []));
/** Selected tracks (stored per experiment); without a selection, the reward tracks. @ai-generated */
const shownTracks = computed<Track[]>(() => {
    const stored = tracksStore.forExperiment(exp.value);
    if (!stored.length) return allTracks.value.flatMap((n) => (n.type === "track" && n.label.startsWith("Rewards") ? [n] : []));
    return stored.flatMap((c) => {
        const t = findTrack(allTracks.value, c.label);
        return t ? [{ ...t, kind: c.kind }] : [];
    });
});
const usingDefault = computed(() => !tracksStore.forExperiment(exp.value).length);

/** Change a shown track's kind (persisting the default selection first if needed). @ai-generated */
function setKind(t: Track, kind: TrackKind): void {
    if (usingDefault.value)
        tracksStore.set(
            exp.value,
            shownTracks.value.map((x) => ({ label: x.label, kind: x.label === t.label ? kind : x.kind })),
        );
    else tracksStore.update(exp.value, { label: t.label, kind });
}
function removeTrack(t: Track): void {
    if (usingDefault.value)
        tracksStore.set(
            exp.value,
            shownTracks.value.filter((x) => x.label !== t.label).map((x) => ({ label: x.label, kind: x.kind })),
        );
    else tracksStore.remove(exp.value, t.label);
}
function moveTrack(i: number, j: number): void {
    if (usingDefault.value)
        tracksStore.set(
            exp.value,
            shownTracks.value.map((x) => ({ label: x.label, kind: x.kind })),
        );
    tracksStore.swap(exp.value, i, j);
}

const isEditable = (el: EventTarget | null) => el instanceof HTMLElement && el.matches("input, textarea, select, [contenteditable='true']");
/** Keyboard navigation of the replay (only when the viewer has focus). @ai-generated */
function onKey(ev: KeyboardEvent): void {
    if (isEditable(ev.target) || !replay.value) return;
    const k = ev.key;
    if (k === " ") store.togglePlay();
    else if (k === "ArrowLeft" || k === "ArrowUp") store.seek(store.t - 1);
    else if (k === "ArrowRight" || k === "ArrowDown") store.seek(store.t + 1);
    else if (k === "Home") store.seek(0);
    else if (k === "End") store.seek(store.maxT);
    else return;
    ev.preventDefault();
}
function onStepInput(ev: Event): void {
    const v = Number.parseInt((ev.target as HTMLInputElement).value, 10);
    if (!Number.isNaN(v)) store.seek(v);
    (ev.target as HTMLInputElement).value = String(store.t);
}
const seedOf = computed(() => episode.value?.seed ?? null);
const kindLabel: Record<string, string> = {
    CombinedReplayAgent: "policy re-run, checked against saved actions",
    ReplayActionsOnlyAgent: "saved actions only",
    SimpleReplayAgent: "policy re-run",
    UNKNOWN: "unknown replay kind",
};
</script>

<template>
    <div class="viewer" tabindex="0" aria-label="Episode replay" @keydown="onKey">
        <template v-if="!store.selected">
            <ReplayUnavailable v-if="store.canReplay === false" :issue="store.blockingIssue" />
            <div class="empty">
                <Icon name="film" :size="26" />
                <p>{{ store.canReplay === false ? "Pick an episode to see its metrics." : "Pick an episode to replay it." }}</p>
            </div>
        </template>

        <template v-else>
            <header class="vhead">
                <div>
                    <div class="eyebrow">Replay</div>
                    <h3>
                        {{ store.selected.run.split("/").pop() }}<span v-if="seedOf !== null" class="muted"> · seed {{ seedOf }}</span> ·
                        test #{{ store.selected.test }}
                        <span class="muted">at step {{ store.selected.step.toLocaleString("en-US") }}</span>
                    </h3>
                </div>
                <span class="sp" />
                <span v-if="replay" class="chip" v-tip="`Replay agent: ${replay.replay_kind}`">{{ kindLabel[replay.replay_kind] }}</span>
                <span
                    class="chip"
                    v-tip="
                        store.replayRule.source === 'trainer'
                            ? `Trainer rule “${store.replayRule.key}” (Settings)`
                            : 'Global replay setting (Settings)'
                    "
                    >only saved actions: {{ store.replayRule.onlySavedActions ? "on" : "off" }}</span
                >
            </header>

            <div v-if="episode" class="tiles" aria-label="Episode metrics">
                <span v-for="(v, k) in episode.metrics" :key="k" class="tile"
                    ><span class="muted">{{ k }}</span
                    ><b>{{ formatNumber(v) }}</b></span
                >
            </div>

            <!-- capability gating -->
            <ReplayUnavailable v-if="gated" :issue="store.blockingIssue" />

            <div v-else-if="store.replayStatus === 'loading'" class="loading">
                <span class="spin" />Replaying the episode… (this re-runs the policy and can take a few seconds)
            </div>

            <div v-else-if="store.replayStatus === 'error'" class="callout err" role="alert">
                <Icon name="error" :size="18" />
                <div>
                    <b>Failed to load the replay</b>
                    <p>{{ store.replayError?.message }}</p>
                    <pre v-if="store.replayError?.issue?.detail">{{ store.replayError.issue.detail }}</pre>
                    <button type="button" class="btn small" @click="store.loadReplay()">
                        <Icon name="restart" :size="13" />Retry replay
                    </button>
                </div>
            </div>

            <template v-else-if="replay">
                <details v-if="replay.replay_mismatch" class="callout warn mismatch">
                    <summary><Icon name="warning" :size="15" /><b>Replay mismatch detected</b></summary>
                    <p>
                        The agent's actions during replay do not match the actions stored on disk during training and testing. The replay
                        below sticks to the actions stored on disk but still shows the "extras" of the loaded agent.
                    </p>
                    <p>
                        <b>Common cause:</b> training may have been performed on GPU but replayed on CPU. CPU and GPU random number
                        generators can produce different sampling sequences and therefore divergent trajectories.
                    </p>
                    <h5>Mismatch details</h5>
                    <ul v-if="replay.mismatch_details.length">
                        <li v-for="(d, i) in replay.mismatch_details" :key="i">{{ d }}</li>
                    </ul>
                    <p v-else class="muted">No mismatch details were provided.</p>
                </details>

                <section class="top">
                    <div class="stage">
                        <div class="frame">
                            <img v-if="frame" :src="frame" :alt="`Frame ${store.t}`" />
                            <span v-else class="muted">No frame</span>
                        </div>
                        <div class="controls">
                            <button
                                type="button"
                                class="play"
                                :aria-label="store.playing ? 'Pause' : 'Play'"
                                v-tip="store.playing ? 'Pause (Space)' : 'Play (Space)'"
                                @click="store.togglePlay()"
                            >
                                <Icon :name="store.playing ? 'pause' : 'play'" :size="14" />
                            </button>
                            <input
                                type="range"
                                min="0"
                                :max="store.maxT"
                                step="1"
                                :value="store.t"
                                aria-label="Replay time step"
                                @input="store.seek(Number(($event.target as HTMLInputElement).value))"
                            />
                            <label class="counter">
                                <span class="sr-only">Step</span>
                                <input :value="store.t" size="3" aria-label="Step" @change="onStepInput" @keydown.enter="onStepInput" />
                                <span class="muted">/ {{ store.maxT }}</span>
                            </label>
                            <select v-model.number="store.fps" aria-label="Playback speed" v-tip="'Frames per second'">
                                <option v-for="s in PLAYBACK_SPEEDS" :key="s" :value="s">{{ s }} fps</option>
                            </select>
                        </div>
                    </div>
                    <ActionPanel :replay="replay" :t="store.t" />
                </section>

                <section class="tracks" aria-label="Timeline tracks">
                    <div class="tbar">
                        <span class="lbl">Timeline</span>
                        <span class="muted small">{{ length }} transitions · click a track to jump</span>
                        <span class="sp" />
                        <span class="muted small"
                            >{{ shownTracks.length }} track{{ shownTracks.length === 1 ? "" : "s"
                            }}{{ usingDefault ? " (default)" : "" }}</span
                        >
                        <button type="button" class="btn small" @click="wizardOpen = true">
                            <Icon name="sliders" :size="13" />Choose tracks
                        </button>
                    </div>
                    <div v-for="(tr, i) in shownTracks" :key="tr.label" class="trow">
                        <div class="tctl">
                            <button type="button" class="hbtn danger" aria-label="Remove track" v-tip="'Remove'" @click="removeTrack(tr)">
                                <Icon name="x" :size="13" />
                            </button>
                            <button type="button" class="hbtn" aria-label="Move up" :disabled="i === 0" @click="moveTrack(i, i - 1)">
                                <Icon name="chevron-up" :size="13" />
                            </button>
                            <button
                                type="button"
                                class="hbtn"
                                aria-label="Move down"
                                :disabled="i === shownTracks.length - 1"
                                @click="moveTrack(i, i + 1)"
                            >
                                <Icon name="chevron-down" :size="13" />
                            </button>
                            <span class="tlabel" :title="tr.label">{{ tr.label }}</span>
                            <select
                                :value="tr.kind"
                                :aria-label="`${tr.label} representation`"
                                @change="setKind(tr, ($event.target as HTMLSelectElement).value as TrackKind)"
                            >
                                <option value="numeric">Numerical</option>
                                <option value="categorical">Categorical</option>
                            </select>
                            <span class="tval">{{
                                formatNumber(tr.values[Math.max(0, Math.min(tr.values.length - 1, store.t - 1))])
                            }}</span>
                        </div>
                        <TimelineTrack
                            :key="`${tr.label}:${tr.kind}`"
                            :track="tr"
                            :t="store.t"
                            :max-t="store.maxT"
                            @select-step="store.seek"
                        />
                    </div>
                </section>

                <details class="agents">
                    <summary>
                        Agent-wise information <span class="muted">({{ agents }} agent{{ agents === 1 ? "" : "s" }})</span>
                    </summary>
                    <div class="agrid">
                        <AgentInfo v-for="a in agents" :key="a" :replay="replay" :agent="a - 1" :t="store.t" />
                    </div>
                </details>

                <TrackWizard v-model:open="wizardOpen" :experiment="exp" :tracks="allTracks" />
            </template>
        </template>
    </div>
</template>

<style scoped>
.viewer {
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-width: 0;
    outline: none;
}
.empty {
    display: grid;
    place-items: center;
    gap: 4px;
    color: var(--ink3);
    padding: 80px 10px;
}
.vhead {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    flex-wrap: wrap;
}
.vhead h3 {
    margin: 2px 0 0;
    font-size: 15px;
}
.sp {
    flex: 1;
}
.tiles {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
}
.tile {
    display: inline-flex;
    gap: 5px;
    font-size: var(--fs-xs);
    padding: 1px 7px;
    border-radius: 6px;
    background: var(--seg-bg);
    font-variant-numeric: tabular-nums;
}
.callout {
    display: flex;
    gap: 10px;
    border-radius: var(--r-sm);
    padding: 10px 12px;
    font-size: var(--fs-sm);
}
.callout p {
    margin: 4px 0;
}
.callout.err {
    background: var(--err-soft);
    color: var(--ink);
}
.callout.err > .icon {
    color: var(--err);
    margin-top: 1px;
}
.callout.err pre {
    white-space: pre-wrap;
    font-size: var(--fs-xs);
    max-height: 10rem;
    overflow: auto;
    margin: 4px 0;
}
.callout.warn {
    background: var(--warn-soft);
    display: block;
}
.mismatch summary {
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--warn);
}
.mismatch h5 {
    margin: 8px 0 2px;
}
.mismatch ul {
    margin: 0;
    padding-left: 18px;
}

.loading {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--ink2);
    padding: 40px 0;
    justify-content: center;
}
.spin {
    width: 16px;
    height: 16px;
    border: 2px solid var(--acc-soft2);
    border-top-color: var(--acc);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
@keyframes spin {
    to {
        transform: rotate(360deg);
    }
}
.top {
    display: grid;
    grid-template-columns: minmax(220px, 400px) minmax(260px, 1fr);
    gap: 12px;
    align-items: start;
}
.stage {
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-width: 0;
}
.frame {
    background: #1e1e26;
    border-radius: var(--r-sm);
    display: grid;
    place-items: center;
    aspect-ratio: 1;
    max-height: 340px;
    overflow: hidden;
}
.frame img {
    max-width: 100%;
    max-height: 340px;
    image-rendering: pixelated;
    display: block;
}
.controls {
    display: flex;
    align-items: center;
    gap: 8px;
}
.controls input[type="range"] {
    flex: 1;
    accent-color: var(--acc);
    min-width: 60px;
}
.play {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    border: 0;
    background: var(--acc);
    color: #fff;
    display: grid;
    place-items: center;
    flex: none;
}
.counter {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    font-variant-numeric: tabular-nums;
    font-size: var(--fs-sm);
}
.counter input {
    width: 3.4em;
    text-align: right;
    border: 1px solid var(--line2);
    border-radius: 6px;
    padding: 1px 4px;
}
select {
    border: 1px solid var(--line2);
    border-radius: 7px;
    background: var(--card);
    padding: 1px 4px;
    font-size: var(--fs-xs);
}
.tracks {
    display: flex;
    flex-direction: column;
    gap: 6px;
    border-top: 1px solid var(--line);
    padding-top: 8px;
}
.tbar {
    display: flex;
    align-items: center;
    gap: 8px;
}
.lbl {
    font-size: var(--fs-eyebrow);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink3);
    font-weight: 600;
}
.small {
    font-size: var(--fs-xs);
}
.trow {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.tctl {
    display: flex;
    align-items: center;
    gap: 2px;
}
.tctl .hbtn {
    width: 22px;
    height: 22px;
}
.tlabel {
    font-size: var(--fs-sm);
    font-weight: 600;
    margin: 0 6px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 50%;
}
.tval {
    margin-left: auto;
    font-size: var(--fs-xs);
    font-variant-numeric: tabular-nums;
    color: var(--ink2);
}
.agents {
    border-top: 1px solid var(--line);
    padding-top: 8px;
}
.agents summary {
    cursor: pointer;
    font-weight: 600;
    color: var(--ink2);
}
.agrid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
    gap: 8px;
    margin-top: 8px;
}
@media (max-width: 1100px) {
    .top {
        grid-template-columns: 1fr;
    }
}
</style>
