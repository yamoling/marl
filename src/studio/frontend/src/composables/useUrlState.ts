/**
 * Mirror the drawer and diff overlay in the URL query (`?exp=<id>&tab=params`, `?diff=1`), so a
 * reload or a shared link restores them, and back/forward navigation updates the UI.
 */
import { watch } from "vue";
import { useRoute, useRouter, type LocationQuery } from "vue-router";
import { isDrawerTab, useUiStore } from "../stores/ui";

/** Query derived from the UI state, keeping unrelated keys. @ai-generated */
export function queryFromUi(current: LocationQuery, s: { drawerId: string | null; drawerTab: string; diffOpen: boolean }): LocationQuery {
  const q: LocationQuery = { ...current };
  delete q.exp;
  delete q.tab;
  delete q.diff;
  if (s.drawerId) {
    q.exp = s.drawerId;
    if (s.drawerTab !== "overview") q.tab = s.drawerTab;
  }
  if (s.diffOpen) q.diff = "1";
  return q;
}

/** @ai-generated */
export function useUrlState(): void {
  const route = useRoute();
  const router = useRouter();
  const ui = useUiStore();

  const apply = (q: LocationQuery) => {
    const exp = typeof q.exp === "string" && q.exp ? q.exp : null;
    if (exp !== ui.drawerId) ui.drawerId = exp;
    const tab = isDrawerTab(q.tab) ? q.tab : "overview";
    if (exp && tab !== ui.drawerTab) ui.drawerTab = tab;
    const diff = q.diff === "1";
    if (diff !== ui.diffOpen) ui.diffOpen = diff;
  };
  apply(route.query);

  watch(
    () => [ui.drawerId, ui.drawerTab, ui.diffOpen] as const,
    () => {
      const next = queryFromUi(route.query, ui);
      if (JSON.stringify(next) !== JSON.stringify(route.query)) void router.replace({ query: next });
    },
  );
  watch(() => route.query, apply);
}
