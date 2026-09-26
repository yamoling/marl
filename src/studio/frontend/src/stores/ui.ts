/**
 * Transient UI state shared across components: experiment drawer (id + tab, mirrored in the
 * URL query by `useUrlState`), parameter diff overlay, start-runs and rename dialogs, and the
 * focused plot (target of "Colour plots by this parameter").
 */
import { defineStore } from "pinia";
import { ref } from "vue";

export const DRAWER_TABS = ["overview", "params", "runs", "issues"] as const;
export type DrawerTab = (typeof DRAWER_TABS)[number];
export const isDrawerTab = (t: unknown): t is DrawerTab => typeof t === "string" && (DRAWER_TABS as readonly string[]).includes(t);

export const useUiStore = defineStore("ui", () => {
  const drawerId = ref<string | null>(null);
  const drawerTab = ref<DrawerTab>("overview");
  const diffOpen = ref(false);
  const launchFor = ref<string | null>(null);
  const renameFor = ref<string | null>(null);
  const focusedPlotId = ref<string | null>(null);

  function openDrawer(id: string, tab: DrawerTab = "overview"): void {
    drawerId.value = id;
    drawerTab.value = tab;
  }
  function closeDrawer(): void {
    drawerId.value = null;
  }

  return { drawerId, drawerTab, diffOpen, launchFor, renameFor, focusedPlotId, openDrawer, closeDrawer };
});
