<script setup lang="ts">
/**
 * Library card: checkbox, name, algorithm, health badge, running indicator, environment, runs,
 * creation date, and a preview sparkline fetched when the card scrolls into view.
 */
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import type { ExperimentSummary } from "../../api";
import Sparkline from "../../charts/Sparkline.vue";
import { fmtPercent, fmtRelativeDate } from "../../domain/format";
import { useLibraryStore } from "../../stores/library";
import HealthBadge from "../shell/HealthBadge.vue";
import ProgressBar from "../shell/ProgressBar.vue";
import PulseDot from "../shell/PulseDot.vue";

const props = defineProps<{ exp: ExperimentSummary; loaded: boolean; selected: boolean; colour?: string }>();
const emit = defineEmits<{ toggle: [id: string]; open: [id: string] }>();
const library = useLibraryStore();
const root = ref<HTMLElement | null>(null);
let io: IntersectionObserver | null = null;

const preview = computed(() => library.previews[props.exp.id]);
const result = computed(() => preview.value?.preview?.result ?? null);
const metricLabel = computed(() => {
    const m = preview.value?.preview?.metric;
    return m ? `${m.table}/${m.metric} mean` : preview.value?.status === "loading" || !preview.value ? "" : "no test metric";
});
const running = computed(() => props.exp.status === "RUNNING");

onMounted(() => {
    if (typeof IntersectionObserver === "undefined") {
        void library.requestPreview(props.exp.id);
        return;
    }
    io = new IntersectionObserver(
        (entries) => {
            if (entries.some((e) => e.isIntersecting)) {
                void library.requestPreview(props.exp.id);
                io?.disconnect();
            }
        },
        { rootMargin: "120px" },
    );
    if (root.value) io.observe(root.value);
});
onBeforeUnmount(() => io?.disconnect());
</script>

<template>
    <label ref="root" class="libcard" :class="{ loaded, sel: selected }">
        <input
            type="checkbox"
            :checked="loaded || selected"
            :disabled="loaded"
            :aria-label="`Select ${exp.id}`"
            @change="emit('toggle', exp.id)"
        />
        <div class="body">
            <div class="lc-top">
                <span class="lc-name" :title="exp.id">{{ exp.name }}</span>
                <span class="algo">{{ exp.algo ?? "?" }}</span>
                <HealthBadge :health="exp.health" :counts="exp.issue_counts" />
                <span v-if="running" class="hb run"><PulseDot :size="6" /> running {{ fmtPercent(exp.progress) }}</span>
                <span v-if="loaded" class="tag">loaded</span>
                <button type="button" class="open" :aria-label="`Open details of ${exp.id}`" @click.prevent.stop="emit('open', exp.id)">
                    Open
                </button>
            </div>
            <div class="lc-sub">
                <span v-if="exp.id !== exp.name" class="mono">{{ exp.id }} · </span>{{ exp.env ?? "unknown environment" }} ·
                {{ exp.n_runs }} run{{ exp.n_runs === 1 ? "" : "s" }} · created
                {{ fmtRelativeDate(exp.created) }}
            </div>
            <ProgressBar v-if="running" class="lc-prog" :value="exp.progress" :height="4" />
        </div>
        <div class="lc-spark">
            <Sparkline
                v-if="result && result.center"
                :x="result.x"
                :y="result.center"
                :lo="result.lo"
                :hi="result.hi"
                :color="colour ?? '#6d5ef5'"
                :width="140"
                :height="34"
            />
            <div v-else class="spark-ph" :class="{ loading: !preview || preview.status === 'loading' }" />
            <small>{{ metricLabel }}</small>
        </div>
    </label>
</template>

<style scoped>
.libcard {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto;
    gap: 14px;
    align-items: center;
    padding: 12px 14px;
    border: 1px solid var(--line);
    border-radius: 12px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: 0.15s;
    background: var(--card);
}
.libcard:hover {
    border-color: #d4d0f3;
    box-shadow: var(--sh);
    transform: translateY(-1px);
}
.libcard.sel {
    border-color: var(--acc);
    background: #fbfaff;
    box-shadow: 0 0 0 3px var(--acc-soft);
}
.libcard.loaded {
    cursor: default;
    background: #fafafa;
}
input {
    width: 17px;
    height: 17px;
    accent-color: var(--acc);
    margin: 0;
}
.lc-top {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}
.lc-name {
    font-weight: 650;
    font-size: 14px;
}
.lc-sub {
    color: var(--ink3);
    font-size: 12px;
    margin-top: 2px;
}
.lc-prog {
    margin-top: 6px;
    max-width: 260px;
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
    font-size: 10.5px;
    font-weight: 600;
    color: var(--acc2);
    background: var(--acc-soft);
    border-radius: 6px;
    padding: 1px 7px;
}
.hb.run {
    display: inline-flex;
    align-items: center;
    gap: 5px;
}
.open {
    border: 1px solid var(--line2);
    background: var(--card);
    border-radius: 6px;
    padding: 0 8px;
    font-size: 11px;
    color: var(--ink2);
    opacity: 0;
    transition: opacity 0.12s;
}
.libcard:hover .open,
.open:focus-visible {
    opacity: 1;
}
.open:hover {
    border-color: var(--acc);
    color: var(--acc2);
}
.lc-spark {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 1px;
    min-width: 140px;
}
.lc-spark small {
    font-size: 10px;
    color: var(--ink3);
}
.spark-ph {
    width: 140px;
    height: 34px;
    border-radius: 6px;
    background: #f6f6f9;
}
.spark-ph.loading {
    background: linear-gradient(90deg, #f3f3f7, #fafafd, #f3f3f7);
    background-size: 200% 100%;
    animation: shimmer 1.6s linear infinite;
}
</style>
