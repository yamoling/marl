/**
 * Run and experiment management actions with confirmation dialogs and toasts: stop a run, stop
 * all runs, restart a run, rename (dialog) and delete (typed confirmation).
 */
import { ApiError } from "../api";
import { useConfirmStore } from "../stores/confirm";
import { useExperimentsStore } from "../stores/experiments";
import { useToasts } from "../stores/toasts";
import { useUiStore } from "../stores/ui";

/** Message of an API error, preferring the attached issue's detail when useful. */
export function errorText(e: unknown): string {
  if (e instanceof ApiError) return e.issue?.detail ? `${e.message} — ${e.issue.detail}` : e.message;
  return (e as Error)?.message ?? String(e);
}

/** @ai-generated */
export function useRunActions() {
  const confirm = useConfirmStore();
  const experiments = useExperimentsStore();
  const toasts = useToasts();
  const ui = useUiStore();
  const short = (runId: string) => runId.split("/").pop() ?? runId;

  async function stopRun(experimentId: string, runId: string): Promise<void> {
    const ok = await confirm.ask({
      title: `Stop ${short(runId)}?`,
      message: `The run of ${experimentId} is interrupted (SIGINT) and marked as cancelled. Its logs are kept and it can be restarted later.`,
      confirmLabel: "Stop run",
      danger: true,
      action: () => experiments.stopRun(experimentId, runId).catch((e) => Promise.reject(new Error(errorText(e)))),
    });
    if (ok) toasts.push({ message: `Stopping ${short(runId)}…` });
  }

  async function stopAll(experimentId: string, nRunning: number): Promise<void> {
    const ok = await confirm.ask({
      title: `Stop all runs of ${experimentId}?`,
      message: `${nRunning} running run${nRunning === 1 ? "" : "s"} will be interrupted and marked as cancelled.`,
      confirmLabel: "Stop all",
      danger: true,
      action: () => experiments.stopExperiment(experimentId).catch((e) => Promise.reject(new Error(errorText(e)))),
    });
    if (ok) toasts.push({ message: `Stopping the runs of ${experiments.name(experimentId)}…` });
  }

  async function restartRun(experimentId: string, runId: string): Promise<void> {
    const ok = await confirm.ask({
      title: `Restart ${short(runId)}?`,
      message: "The run restarts from its latest checkpoint on an automatically chosen device.",
      confirmLabel: "Restart",
      action: () => experiments.restartRun(experimentId, runId).catch((e) => Promise.reject(new Error(errorText(e)))),
    });
    if (ok) toasts.push({ level: "success", message: `Restarted ${short(runId)}` });
  }

  function rename(experimentId: string): void {
    ui.renameFor = experimentId;
  }

  /** Typed confirmation (the experiment's folder name), then delete, unload and clean plots. @ai-generated */
  async function remove(experimentId: string): Promise<void> {
    const name = experimentId.split("/").pop() ?? experimentId;
    const ok = await confirm.ask({
      title: `Delete ${experimentId}?`,
      message: "This permanently deletes the experiment folder with all its runs, metrics and weights. It cannot be undone.",
      confirmLabel: "Delete permanently",
      danger: true,
      confirmText: name,
      action: () => experiments.remove(experimentId).catch((e) => Promise.reject(new Error(errorText(e)))),
    });
    if (!ok) return;
    if (ui.drawerId === experimentId) ui.closeDrawer();
    toasts.push({ level: "success", message: `Deleted ${experimentId}; it was removed from the workspace and its plots` });
  }

  return { stopRun, stopAll, restartRun, rename, remove };
}
