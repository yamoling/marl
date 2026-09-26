/**
 * Experiment library: search (`q` is sent to the API), client-side facets with counts,
 * selection, and lazily fetched preview sparklines.
 */
import { defineStore } from "pinia";
import { computed, markRaw, ref, watch } from "vue";
import { useApi, type ExperimentSummary, type Preview } from "../api";
import { statusGroup } from "../domain/status";

export type FacetKey = "algo" | "status" | "health";
export type PreviewEntry = { status: "loading" | "ok" | "error"; preview: Preview | null };

const SEARCH_DEBOUNCE_MS = 200;
export const algoLabel = (s: ExperimentSummary) => s.algo ?? "unknown";

export const useLibraryStore = defineStore("library", () => {
  const open = ref(false);
  const q = ref("");
  const items = ref<ExperimentSummary[]>([]);
  const badCount = ref(0);
  const loading = ref(false);
  const error = ref<string | null>(null);
  const stale = ref(true);
  const facets = ref<Record<FacetKey, string[]>>({ algo: [], status: [], health: [] });
  const selected = ref<string[]>([]);
  const previews = ref<Record<string, PreviewEntry>>({});
  let seq = 0;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const facetValue: Record<FacetKey, (s: ExperimentSummary) => string> = {
    algo: algoLabel,
    status: (s) => statusGroup(s.status),
    health: (s) => s.health,
  };

  /** Facet values with counts over the current search results. @ai-generated */
  const facetCounts = computed(() => {
    const out: Record<FacetKey, [string, number][]> = { algo: [], status: [], health: [] };
    for (const k of Object.keys(out) as FacetKey[]) {
      const m = new Map<string, number>();
      if (k === "status") ["running", "completed", "other"].forEach((v) => m.set(v, 0));
      if (k === "health") ["ok", "warning", "error"].forEach((v) => m.set(v, 0));
      for (const s of items.value) m.set(facetValue[k](s), (m.get(facetValue[k](s)) ?? 0) + 1);
      out[k] = [...m.entries()].sort((a, b) => (k === "algo" ? a[0].localeCompare(b[0]) : 0));
    }
    return out;
  });

  const filtered = computed(() =>
    items.value.filter((s) =>
      (Object.keys(facets.value) as FacetKey[]).every((k) => !facets.value[k].length || facets.value[k].includes(facetValue[k](s))),
    ),
  );

  /** Fetch the list for the current query (latest request wins). @ai-generated */
  async function fetch(): Promise<void> {
    const my = ++seq;
    loading.value = true;
    error.value = null;
    try {
      const r = await useApi().listExperiments({ q: q.value.trim() || undefined });
      if (my !== seq) return;
      items.value = r.items;
      badCount.value = r.bad.length;
      stale.value = false;
    } catch (e) {
      if (my === seq) error.value = (e as Error)?.message ?? String(e);
    } finally {
      if (my === seq) loading.value = false;
    }
  }

  watch(q, () => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => void fetch(), SEARCH_DEBOUNCE_MS);
  });

  function show(): void {
    open.value = true;
    selected.value = [];
    if (stale.value || error.value) void fetch();
  }

  function markStale(): void {
    stale.value = true;
    if (open.value) void fetch();
  }

  function toggleFacet(k: FacetKey, v: string): void {
    const cur = facets.value[k];
    facets.value = { ...facets.value, [k]: cur.includes(v) ? cur.filter((x) => x !== v) : [...cur, v] };
  }

  function toggleSelected(id: string): void {
    selected.value = selected.value.includes(id) ? selected.value.filter((x) => x !== id) : [...selected.value, id];
  }

  /** Fetch a card's preview once (called when the card scrolls into view). @ai-generated */
  async function requestPreview(id: string): Promise<void> {
    if (previews.value[id]) return;
    previews.value = { ...previews.value, [id]: { status: "loading", preview: null } };
    try {
      const p = await useApi().getPreview(id, 60);
      previews.value = { ...previews.value, [id]: { status: "ok", preview: markRaw(p) } };
    } catch {
      previews.value = { ...previews.value, [id]: { status: "error", preview: null } };
    }
  }

  return {
    open,
    q,
    items,
    badCount,
    loading,
    error,
    stale,
    facets,
    selected,
    previews,
    facetCounts,
    filtered,
    fetch,
    show,
    markStale,
    toggleFacet,
    toggleSelected,
    requestPreview,
  };
});
