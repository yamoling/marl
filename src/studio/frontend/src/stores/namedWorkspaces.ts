/** Backend workspace metadata and the active plotting workspace. */
import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { useApi, type NamedWorkspace } from "../api";
import { useExperimentsStore } from "./experiments";
import { useLibraryStore } from "./library";
import { useLiveStore } from "./live";
import { useReplayStore } from "./replay";
import { useSeriesStore } from "./series";
import { useUiStore } from "./ui";
import { useWorkspaceStore } from "./workspace";
import { STORAGE_KEY } from "../domain/workspace";
import { removeStorage } from "./storage";

export const useNamedWorkspacesStore = defineStore("namedWorkspaces", () => {
  const workspaces = ref<NamedWorkspace[]>([]);
  const selected = ref<string | null>(null);
  /** Only a deliberate home-screen entry sets activeId; a server selection does not skip home. */
  const activeId = ref<string | null>(null);
  const switching = ref(false);
  const active = computed(() => workspaces.value.find((w) => w.id === activeId.value) ?? null);
  let legacyId: string | null = null;

  /** Refresh names and selected workspace from the server. @ai-generated */
  async function refresh(): Promise<void> {
    const result = await useApi().listWorkspaces();
    if (!legacyId) legacyId = result.selected;
    selected.value = result.selected;
    workspaces.value = result.workspaces;
  }

  /** Replace a workspace snapshot without disturbing other entries or their order. @ai-generated */
  function update(workspace: NamedWorkspace): void {
    const index = workspaces.value.findIndex((w) => w.id === workspace.id);
    if (index < 0) workspaces.value.push(workspace);
    else workspaces.value[index] = workspace;
  }

  /** Create a workspace; entering it remains an explicit home-screen action. @ai-generated */
  async function create(name: string, logdir?: string): Promise<NamedWorkspace> {
    const workspace = await useApi().createWorkspace(name.trim(), logdir?.trim());
    update(workspace);
    return workspace;
  }

  /** Rename a workspace by its stable id so plots remain associated with it. @ai-generated */
  async function rename(id: string, name: string): Promise<void> {
    update(await useApi().renameWorkspace(id, name.trim()));
  }

  /** Save a root logdir; active experiment data must be reloaded from the new root. @ai-generated */
  async function setLogdir(id: string, logdir: string): Promise<void> {
    switching.value = true;
    try {
      const workspace = await useApi().setWorkspaceLogdir(id, logdir.trim());
      if (activeId.value === id) {
        useLiveStore().disconnect();
        useExperimentsStore().clear();
        useLibraryStore().clear();
        useSeriesStore().clear();
        useReplayStore().reset();
        useExperimentsStore().ensureAll();
        useLiveStore().connect();
      }
      update(workspace);
    } finally {
      switching.value = false;
    }
  }

  /** Trash the workspace layout and metadata, leaving all experiment files untouched. @ai-generated */
  async function trash(id: string): Promise<void> {
    const result = await useApi().deleteWorkspace(id);
    if (activeId.value === id) {
      useLiveStore().disconnect();
      useExperimentsStore().clear();
      useLibraryStore().clear();
      useSeriesStore().clear();
      useReplayStore().reset();
      useWorkspaceStore().switchTo(result.selected ?? "");
      activeId.value = null;
    }
    removeStorage(`${STORAGE_KEY}.${encodeURIComponent(id)}`);
    if (legacyId === id) legacyId = null;
    selected.value = result.selected;
    workspaces.value = result.workspaces;
  }

  /** Select on the server before swapping plots; failed selects leave the current view intact. @ai-generated */
  async function enter(id: string): Promise<void> {
    switching.value = true;
    try {
      const workspace = await useApi().selectWorkspace(id);
      useLiveStore().disconnect();
      useExperimentsStore().clear();
      useLibraryStore().clear();
      useSeriesStore().clear();
      useReplayStore().reset();
      if (activeId.value !== id) {
        const ui = useUiStore();
        ui.closeDrawer();
        ui.diffOpen = false;
        ui.focusedPlotId = null;
        ui.launchFor = null;
        ui.renameFor = null;
        useWorkspaceStore().switchTo(id, legacyId === id);
      }
      update(workspace);
      selected.value = id;
      activeId.value = id;
    } finally {
      switching.value = false;
    }
  }

  return { workspaces, selected, activeId, active, switching, refresh, create, rename, setLogdir, trash, enter };
});
