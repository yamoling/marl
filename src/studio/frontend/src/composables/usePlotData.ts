/**
 * Everything a plot card shows, derived from its spec: expansion (requests, legend, notes),
 * series outcomes from the cache (fetched on demand, refetched on invalidation) and the
 * presentation (chart series, notes).
 */
import { computed, watch, type Ref } from "vue";
import { expand, type Expansion, type PlotSpec, type Presentation } from "../domain/plot";
import { useExperimentsStore } from "../stores/experiments";
import { useSeriesStore } from "../stores/series";

/** @ai-generated */
export function usePlotData(plot: Ref<PlotSpec>) {
  const experiments = useExperimentsStore();
  const series = useSeriesStore();

  const expansion = computed<Expansion>(() => expand(plot.value, experiments.context));

  watch(
    [() => expansion.value.requests, () => series.revision],
    ([requests]) => {
      for (const q of requests) series.ensure(q);
    },
    { immediate: true },
  );

  const entries = computed(() => expansion.value.requests.map((q) => series.get(q)));
  const presentation = computed<Presentation>(() => expansion.value.present(entries.value.map((e) => e?.outcome ?? undefined)));
  const loading = computed(() => entries.value.some((e) => !e || e.status === "loading"));
  const revalidating = computed(() => entries.value.some((e) => e?.revalidating));

  return { expansion, presentation, loading, revalidating };
}
