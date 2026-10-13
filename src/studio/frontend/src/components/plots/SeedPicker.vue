<script setup lang="ts">
/**
 * Seed picker (per experiment, all by default). `seeds` follows `PlotSpec.runs.seeds`: `null` or a
 * missing key means all runs of that experiment.
 */
import { computed, ref } from "vue";
import Icon from "../shell/Icon.vue";
import Popover from "../shell/Popover.vue";
import type { SeedExperiment } from "./types";

const props = defineProps<{ seeds: Record<string, number[]> | null; experiments: SeedExperiment[] }>();
const emit = defineEmits<{ "update:seeds": [v: Record<string, number[]> | null] }>();
const btn = ref<HTMLButtonElement | null>(null);
const open = ref(false);

const seedsOf = (e: SeedExperiment) => e.runs.map((r) => r.seed).filter((s): s is number => s !== null);
const selected = (e: SeedExperiment) => props.seeds?.[e.id] ?? seedsOf(e);

const label = computed(() => {
    if (!props.seeds || !Object.keys(props.seeds).length) return "all seeds";
    let total = 0;
    let shown = 0;
    for (const e of props.experiments) {
        total += seedsOf(e).length;
        shown += selected(e).filter((s) => seedsOf(e).includes(s)).length;
    }
    return `${shown}/${total} runs`;
});

/** Set one experiment's subset; full selection removes its key, no key left → null. @ai-generated */
function set(e: SeedExperiment, list: number[] | null): void {
    const next = { ...(props.seeds ?? {}) };
    const all = seedsOf(e);
    if (list === null || (list.length === all.length && all.every((s) => list.includes(s)))) delete next[e.id];
    else next[e.id] = [...list].sort((a, b) => a - b);
    emit("update:seeds", Object.keys(next).length ? next : null);
}
function toggle(e: SeedExperiment, seed: number): void {
    const cur = selected(e);
    set(e, cur.includes(seed) ? cur.filter((s) => s !== seed) : [...cur, seed]);
}
</script>

<template>
    <button ref="btn" type="button" class="addbtn" data-act="seeds" :aria-expanded="open" @click="open = !open">
        {{ label }} <Icon name="chevron-down" :size="11" />
    </button>
    <Popover v-model:open="open" :anchor="btn" :min-width="240" label="Seeds shown">
        <h4>Seeds shown</h4>
        <button type="button" class="mi" :disabled="!seeds" @click="emit('update:seeds', null)">
            <Icon name="restart" :size="12" />All seeds of all experiments
        </button>
        <template v-for="e in experiments" :key="e.id">
            <hr />
            <label class="mi head">
                <input
                    type="checkbox"
                    :checked="!seeds?.[e.id]"
                    :aria-label="`All seeds of ${e.name}`"
                    @change="set(e, ($event.target as HTMLInputElement).checked ? null : [])"
                />
                <span class="dot" :style="{ '--c': e.colour }" /><b>{{ e.name }}</b
                ><small>{{ selected(e).length }}/{{ seedsOf(e).length }}</small>
            </label>
            <div class="seeds">
                <label v-for="s in seedsOf(e)" :key="s" class="seed">
                    <input type="checkbox" :checked="selected(e).includes(s)" :data-seed="`${e.id}:${s}`" @change="toggle(e, s)" />{{ s }}
                </label>
            </div>
        </template>
        <div v-if="!experiments.length" class="mi muted">No experiment in this plot.</div>
    </Popover>
</template>

<style scoped>
.head {
    font-size: 12px;
}
.seeds {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 10px;
    padding: 2px 10px 6px 32px;
}
.seed {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    font-family: var(--mono);
}
.seed input {
    accent-color: var(--acc);
    margin: 0;
}
</style>
