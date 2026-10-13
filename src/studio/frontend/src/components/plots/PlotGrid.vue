<script setup lang="ts">
/** Plot workspace, including field drop targets and the transient maximized plot. */
import { computed } from "vue";
import { endDrag, readDrop } from "../../composables/dnd";
import { usePlotActions } from "../../composables/usePlotActions";
import { useWorkspaceStore } from "../../stores/workspace";
import Overlay from "../shell/Overlay.vue";
import PlotCard from "./PlotCard.vue";

const workspace = useWorkspaceStore();
const actions = usePlotActions();
const maximized = computed(() => workspace.plots.find((p) => p.id === workspace.maximizedId));
const overlayOpen = computed({
  get: () => !!maximized.value,
  set: (open: boolean) => { if (!open) workspace.maximizedId = null; },
});

/** Apply a field or move a plot to the end of the grid. @ai-generated */
function drop(ev: DragEvent): void {
  const payload = readDrop(ev);
  if (payload?.kind === "plot") workspace.movePlot(payload.id, workspace.plots.length);
  else if (payload) actions.plotFromField(payload);
  endDrag();
}
</script>

<template>
  <div class="plot-grid">
    <template v-for="plot in workspace.plots" :key="plot.id">
      <div v-if="plot.id === workspace.maximizedId" class="placeholder">{{ plot.title }} is maximized</div>
      <PlotCard v-else :plot="plot" />
    </template>
    <div class="new-plot" @dragover.prevent @drop.prevent="drop">
      <button type="button" class="btn" @click="actions.newEmptyPlot()">New plot</button>
      <span class="muted">Drop a metric or parameter here to create a plot</span>
    </div>
  </div>
  <Overlay v-model:open="overlayOpen" :label="maximized?.title ?? ''">
    <PlotCard v-if="maximized" :plot="maximized" maximized />
  </Overlay>
</template>

<style scoped>
.plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 460px), 1fr)); gap: 18px; align-items: start; }
.new-plot, .placeholder { min-height: 120px; border: 1px dashed var(--line2); border-radius: var(--r); display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; padding: 20px; }
.new-plot span { font-size: 12px; text-align: center; }
.placeholder { color: var(--ink3); }
</style>
