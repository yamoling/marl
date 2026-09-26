<script setup lang="ts">
/**
 * Library sheet: search (free text or parameter queries), hint chips, facets with counts, cards
 * with lazy sparklines, multi-select and "Load N selected".
 */
import { computed, nextTick, ref, watch } from "vue";
import { useExperimentsStore } from "../../stores/experiments";
import { useLibraryStore } from "../../stores/library";
import { useToasts } from "../../stores/toasts";
import { useUiStore } from "../../stores/ui";
import Icon from "../shell/Icon.vue";
import Sheet from "../shell/Sheet.vue";
import ExperimentCard from "./ExperimentCard.vue";
import FacetChips from "./FacetChips.vue";
import QueryHints from "./QueryHints.vue";

const library = useLibraryStore();
const experiments = useExperimentsStore();
const toasts = useToasts();
const ui = useUiStore();
const search = ref<HTMLInputElement | null>(null);

const nSel = computed(() => library.selected.length);
const loadedSet = computed(() => new Set(experiments.loaded));

watch(
    () => library.open,
    async (o) => {
        if (!o) return;
        await nextTick();
        search.value?.focus();
        search.value?.select();
    },
);

/** Focus the search box (Ctrl+K while already open). */
function focusSearch(): void {
    search.value?.focus();
    search.value?.select();
}
defineExpose({ focusSearch });

/** Close the library and show the experiment in the drawer (loaded or not). */
function openDetails(id: string): void {
    library.open = false;
    ui.openDrawer(id);
}

function selectAllVisible(): void {
    library.selected = [...new Set([...library.selected, ...library.filtered.filter((e) => !loadedSet.value.has(e.id)).map((e) => e.id)])];
}

/** Load the selection and close the sheet. @ai-generated */
function loadSelected(): void {
    const ids = experiments.load(library.selected);
    library.open = false;
    library.selected = [];
    if (ids.length)
        toasts.push({
            level: "success",
            message: `Loaded ${ids.length} experiment${ids.length > 1 ? "s" : ""}: ${ids.map((id) => experiments.name(id)).join(", ")}`,
        });
}
</script>

<template>
    <Sheet v-model:open="library.open" variant="large" title="Add experiments" eyebrow="Library">
        <template #header>
            <div class="eyebrow">Library</div>
            <h2>Add experiments</h2>
            <label class="search">
                <Icon name="search" :size="15" class="muted" />
                <input
                    ref="search"
                    v-model="library.q"
                    placeholder="Search by name, algorithm, or a parameter query…"
                    autocomplete="off"
                    spellcheck="false"
                    aria-label="Search experiments"
                />
                <button v-if="library.q" class="hbtn" aria-label="Clear search" @click="library.q = ''">
                    <Icon name="x" :size="13" />
                </button>
            </label>
            <QueryHints @pick="(q) => (library.q = q)" />
            <FacetChips :counts="library.facetCounts" :active="library.facets" @toggle="library.toggleFacet" />
        </template>

        <div v-if="library.error" class="empty">
            Could not load the library: {{ library.error }}
            <div><button class="btn small" @click="library.fetch()">Retry</button></div>
        </div>
        <template v-else>
            <div v-if="library.badCount" class="bad">
                {{ library.badCount }} entr{{ library.badCount > 1 ? "ies" : "y" }} could not be read and are not shown.
            </div>
            <ExperimentCard
                v-for="e in library.filtered"
                :key="e.id"
                :exp="e"
                :loaded="loadedSet.has(e.id)"
                :selected="library.selected.includes(e.id)"
                :colour="experiments.colours[e.id]"
                @toggle="library.toggleSelected"
                @open="openDetails"
            />
            <div v-if="!library.filtered.length && !library.loading" class="empty">
                No experiment matches<template v-if="library.q"> “{{ library.q }}”</template>.
            </div>
            <div v-if="library.loading && !library.items.length" class="empty">Loading…</div>
        </template>

        <template #footer>
            <span class="muted">{{ library.filtered.length }} of {{ library.items.length }} experiments · {{ nSel }} selected</span>
            <span class="sp" />
            <button class="btn" @click="selectAllVisible">Select all visible</button>
            <button class="btn primary" :disabled="!nSel" @click="loadSelected">Load {{ nSel || "" }} selected</button>
        </template>
    </Sheet>
</template>

<style scoped>
h2 {
    margin: 2px 0 12px;
    font-size: 20px;
    letter-spacing: -0.02em;
}
.search {
    display: flex;
    align-items: center;
    gap: 8px;
    border: 1px solid var(--line2);
    border-radius: 10px;
    padding: 8px 12px;
    background: var(--card);
    transition: 0.15s;
}
.search:focus-within {
    border-color: var(--acc);
    box-shadow: var(--focus-ring);
}
.search input {
    border: 0;
    outline: none;
    flex: 1;
    font-size: 14px;
    background: transparent;
    min-width: 0;
}
.bad {
    color: var(--warn);
    background: var(--warn-soft);
    border-radius: 8px;
    padding: 6px 10px;
    margin-bottom: 8px;
    font-size: var(--fs-sm);
}
.sp {
    flex: 1;
}
</style>
