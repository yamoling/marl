<script setup lang="ts">
/** All shelves of a plot, wired to the workspace store. `vertical` = side column of a maximized plot. */
import { computed } from "vue";
import { metricGroups, paramFields } from "../../domain/fields";
import { plotExperiments, type PlotSpec } from "../../domain/plot";
import { useExperimentsStore } from "../../stores/experiments";
import { useWorkspaceStore } from "../../stores/workspace";
import AdvancedPopover from "./AdvancedPopover.vue";
import ColourByShelf from "./ColourByShelf.vue";
import ExperimentsShelf from "./ExperimentsShelf.vue";
import LineStyleShelf from "./LineStyleShelf.vue";
import RunsShelf from "./RunsShelf.vue";
import StatShelf from "./StatShelf.vue";
import YShelf from "./YShelf.vue";

const props = withDefaults(defineProps<{ plot: PlotSpec; vertical?: boolean }>(), { vertical: false });
const experiments = useExperimentsStore();
const workspace = useWorkspaceStore();
const patch = (p: Partial<PlotSpec>) => workspace.patchPlot(props.plot.id, p);

const metrics = computed(() => metricGroups(experiments.loaded, Object.fromEntries(experiments.loaded.map((id) => [id, experiments.catalog(id)]))));
const params = computed(() => {
  const f = paramFields(experiments.loaded, Object.fromEntries(experiments.loaded.map((id) => [id, experiments.detail(id)?.params])));
  return [...f.differing, ...f.constant];
});
const loaded = computed(() => experiments.loaded.map((id) => ({ id, name: experiments.name(id), colour: experiments.colours[id] })));
const seedExperiments = computed(() =>
  plotExperiments(props.plot, experiments.loaded).map((id) => ({ id, name: experiments.name(id), colour: experiments.colours[id], runs: experiments.runs(id) })),
);
</script>

<template>
  <div class="shelves" :class="{ vertical }">
    <YShelf
      :y="plot.y"
      :line-style-by="plot.lineStyleBy"
      :groups="metrics.groups"
      :n="metrics.n"
      :auto-open="workspace.addYFor === plot.id"
      @opened="workspace.addYFor = null"
      @update:y="(y) => patch({ y })"
    />
    <ExperimentsShelf :experiments="plot.experiments" :loaded="loaded" @update:experiments="(v) => patch({ experiments: v })" />
    <ColourByShelf :colour-by="plot.colourBy" :params="params" :runs-mode="plot.runs.mode" @update:colour-by="(colourBy) => patch({ colourBy })" />
    <LineStyleShelf :value="plot.lineStyleBy" @update="(lineStyleBy) => patch({ lineStyleBy })" />
    <RunsShelf :runs="plot.runs" :experiments="seedExperiments" @update:runs="(runs) => patch({ runs })" />
    <StatShelf :stat="plot.stat" :dim="plot.runs.mode === 'runs'" @update:stat="(stat) => patch({ stat })" />
    <div class="shelf">
      <AdvancedPopover :x="plot.x" :log-y="plot.logY" @update:x="(x) => patch({ x })" @update:log-y="(logY) => patch({ logY })" />
    </div>
  </div>
</template>

<style scoped>
.shelves {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 16px;
  padding: 2px 14px 10px;
  border-bottom: 1px dashed var(--line);
}
.shelves.vertical {
  flex-direction: column;
  flex-wrap: nowrap;
  gap: 14px;
  border-bottom: 0;
  border-right: 1px dashed var(--line);
  overflow: auto;
  padding: 16px;
}
.shelves.vertical :deep(.shelf) {
  flex-direction: column;
  align-items: flex-start;
}
</style>
