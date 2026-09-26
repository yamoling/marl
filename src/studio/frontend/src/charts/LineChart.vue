<script setup lang="ts">
/**
 * Vue wrapper of `render.ts`: mounts the chart, redraws on resize (ResizeObserver, one frame
 * at a time) and on spec changes, and emits `point-click` when `clickable`.
 */
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import { mount, themeFromCSS, type ChartHandle, type ChartSpec } from "./render";

const props = withDefaults(
  defineProps<{
    /** Chart spec without the click callback (use `clickable` + the `point-click` event). */
    spec: Omit<ChartSpec, "onPointClick">;
    clickable?: boolean;
    height?: string;
  }>(),
  { clickable: false, height: "260px" },
);
const emit = defineEmits<{ "point-click": [payload: { x: number; seriesIndex: number; key: string | undefined }] }>();

const root = ref<HTMLDivElement | null>(null);
let handle: ChartHandle | null = null;
let ro: ResizeObserver | null = null;
let frame = 0;

/** Full spec for the renderer: theme from CSS variables and the click bridge. @ai-generated */
function fullSpec(): ChartSpec {
  const s = props.spec;
  return {
    ...s,
    theme: s.theme ?? (root.value ? themeFromCSS(root.value) : undefined),
    onPointClick: props.clickable
      ? (x, seriesIndex) => emit("point-click", { x, seriesIndex, key: s.series[seriesIndex]?.key })
      : null,
  };
}

onMounted(() => {
  if (!root.value) return;
  handle = mount(root.value, fullSpec());
  if (typeof ResizeObserver !== "undefined") {
    ro = new ResizeObserver(() => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => handle?.redraw());
    });
    ro.observe(root.value);
  }
});

watch(
  () => [props.spec, props.clickable] as const,
  () => handle?.update(fullSpec()),
);

onBeforeUnmount(() => {
  cancelAnimationFrame(frame);
  ro?.disconnect();
  handle?.destroy();
  handle = null;
});

defineExpose({
  exportSVG: () => handle?.exportSVG() ?? "",
  exportPNG: (scale?: number) => (handle ? handle.exportPNG(scale) : Promise.reject(new Error("chart not mounted"))),
  resetZoom: () => handle?.resetZoom(),
});
</script>

<template>
  <div ref="root" class="line-chart" :style="{ height }" />
</template>

<style scoped>
.line-chart {
  width: 100%;
  min-width: 0;
  position: relative;
}
</style>
