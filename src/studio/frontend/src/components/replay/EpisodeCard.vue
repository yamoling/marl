<script setup lang="ts">
/**
 * One test episode: test number, score, exit, length and the other scalar metrics. The left edge
 * colour tells success (exit_rate 1), partial success or failure.
 */
import { computed } from "vue";
import type { EpisodeSummary } from "../../api";
import { episodeOutcome, extraMetrics, formatNumber } from "../../domain/replay";

const props = defineProps<{ episode: EpisodeSummary; scoreKey: string | null; selected: boolean }>();
defineEmits<{ select: [] }>();

const outcome = computed(() => episodeOutcome(props.episode.metrics));
const exitLabel = computed(() => {
    const x = props.episode.metrics.exit_rate;
    if (typeof x !== "number") return null;
    return outcome.value === "success" ? "exit ✓" : outcome.value === "partial" ? `exit ${Math.round(x * 100)}%` : "no exit";
});
const score = computed(() => (props.scoreKey ? props.episode.metrics[props.scoreKey] : undefined));
const length = computed(() => props.episode.metrics.episode_len);
const MAX_CHIPS = 4;
const allExtras = computed(() => extraMetrics(props.episode.metrics, props.scoreKey));
const extras = computed(() => allExtras.value.slice(0, MAX_CHIPS));
const hidden = computed(() => allExtras.value.length - extras.value.length);
const title = computed(() =>
    [
        `${props.episode.run} · test #${props.episode.test}`,
        ...Object.entries(props.episode.metrics).map(([k, v]) => `${k}: ${formatNumber(v)}`),
        props.episode.has_actions ? "" : "No saved actions: the replay re-runs the policy",
    ]
        .filter(Boolean)
        .join("\n"),
);
</script>

<template>
    <button
        type="button"
        class="epc"
        :class="[outcome, { sel: selected }]"
        :aria-pressed="selected"
        :title="title"
        @click="$emit('select')"
    >
        <span class="t">
            <span>test #{{ episode.test }}</span>
            <b v-if="score !== undefined">{{ formatNumber(score) }}</b>
        </span>
        <span class="chips">
            <span v-if="exitLabel" class="o">{{ exitLabel }}</span>
            <span v-if="typeof length === 'number'">{{ length }} steps</span>
            <span v-for="[k, v] in extras" :key="k" class="x">{{ k }} {{ formatNumber(v) }}</span>
            <span v-if="hidden > 0" class="x">+{{ hidden }} more</span>
            <span v-if="!episode.has_actions" class="x">no saved actions</span>
        </span>
    </button>
</template>

<style scoped>
.epc {
    --oc: var(--missing);
    text-align: left;
    border: 1px solid var(--line);
    background: var(--card);
    border-radius: 10px;
    padding: 7px 10px 7px 13px;
    position: relative;
    overflow: hidden;
    transition: 0.15s;
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
}
.epc::before {
    content: "";
    position: absolute;
    inset: 0 auto 0 0;
    width: 4px;
    background: var(--oc);
}
.epc.success {
    --oc: var(--ok);
}
.epc.partial {
    --oc: var(--warn);
}
.epc.failure {
    --oc: var(--err);
}
.epc:hover {
    border-color: var(--line-hover);
    transform: translateY(-1px);
    box-shadow: var(--sh);
}
.epc.sel {
    border-color: var(--acc);
    box-shadow: 0 0 0 3px var(--acc-soft);
}
.t {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    font-size: var(--fs-sm);
    color: var(--ink2);
}
.t b {
    color: var(--ink);
    font-variant-numeric: tabular-nums;
}
.chips {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
}
.chips span {
    font-size: 10.5px;
    padding: 0 6px;
    border-radius: 5px;
    background: var(--seg-bg);
    color: var(--ink2);
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
}
.chips .o {
    color: var(--oc);
    background: color-mix(in srgb, var(--oc) 13%, transparent);
    font-weight: 600;
}
.chips .x {
    color: var(--ink3);
}
</style>
