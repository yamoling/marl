<script setup lang="ts">
/** Drop zone creating a plot from a field; a click creates an empty plot. Plots dropped here move to the end. */
import { ref } from "vue";
import { dragging, endDrag, readDrop } from "../../composables/dnd";
import { usePlotActions } from "../../composables/usePlotActions";
import { useWorkspaceStore } from "../../stores/workspace";

const actions = usePlotActions();
const workspace = useWorkspaceStore();
const over = ref(false);

function onDragOver(ev: DragEvent): void {
  if (!dragging.value) return;
  ev.preventDefault();
  over.value = true;
}
function onDrop(ev: DragEvent): void {
  over.value = false;
  const p = readDrop(ev);
  if (!p) return;
  ev.preventDefault();
  if (p.kind === "plot") workspace.movePlot(p.id, workspace.plots.length);
  else actions.plotFromField(p);
  endDrag();
}
</script>

<template>
  <button
    type="button"
    class="newcard"
    :class="{ accepting: dragging && dragging.kind !== 'plot', 'drop-over': over }"
    @click="actions.newEmptyPlot()"
    @dragover="onDragOver"
    @dragleave="over = false"
    @drop="onDrop"
  >
    <div>
      <div class="plus">＋</div>
      <b>Drop a field here to create a plot</b>
      <div class="sub">or click to start an empty one</div>
    </div>
  </button>
</template>

<style scoped>
.newcard {
  border: 2px dashed #d9d6ec;
  border-radius: var(--r);
  min-height: 200px;
  width: 100%;
  display: grid;
  place-items: center;
  text-align: center;
  color: var(--ink3);
  background: transparent;
  transition: 0.15s;
  padding: 24px;
}
.newcard:hover,
.newcard.accepting {
  border-color: var(--acc);
  color: var(--acc2);
}
.newcard.drop-over {
  background: var(--acc-soft);
  transform: scale(1.01);
}
.plus {
  font-size: 30px;
  line-height: 1;
  font-weight: 300;
  margin-bottom: 6px;
}
.sub {
  font-size: 12px;
  margin-top: 4px;
}
</style>
