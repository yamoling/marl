/**
 * Capability summary (Metrics, Parameters, Replay, Launch) with the reason behind each missing
 * capability, and the tooltip explaining why "Start runs" is disabled.
 */
import type { Capabilities, Issue } from "../api/schemas";

export type CapabilityState = "yes" | "no" | "partial" | "checking";
export type CapabilityLine = { key: "metrics" | "params" | "replay" | "launch"; label: string; state: CapabilityState; value: string; reason: string | null };

const LEVEL_RANK = { error: 0, warning: 1, info: 2 } as const;
const REASON_CODES: Record<CapabilityLine["key"], string[]> = {
  metrics: ["missing-table", "unreadable-table", "no-runs", "no-metrics"],
  params: ["missing-experiment-json", "invalid-experiment-json", "missing-keys", "deserialize-failed"],
  replay: ["replay-unavailable", "deserialize-failed", "missing-keys", "missing-experiment-json"],
  launch: ["deserialize-failed", "missing-keys", "missing-experiment-json", "invalid-experiment-json", "not-launchable"],
};

/**
 * The issue best explaining a missing capability: one with a relevant code first, else the most
 * severe experiment-level issue.
 *
 * @ai-generated
 */
export function reasonIssue(issues: Issue[], key: CapabilityLine["key"]): Issue | null {
  const own = issues.filter((i) => !i.scope.startsWith("run:"));
  const byCode = REASON_CODES[key].map((c) => own.find((i) => i.code === c)).find(Boolean);
  if (byCode) return byCode;
  return [...own].filter((i) => i.level !== "info").sort((a, b) => LEVEL_RANK[a.level] - LEVEL_RANK[b.level])[0] ?? null;
}

const describe = (i: Issue | null): string | null => (i ? `${i.message}${i.path ? ` (at ${i.path})` : ""}` : null);

/** @ai-generated */
export function capabilityLines(c: Capabilities, issues: Issue[], checking = false): CapabilityLine[] {
  const bool = (key: "replay" | "launch", label: string): CapabilityLine => {
    const v = c[key];
    if (v === null) return { key, label, state: "checking", value: checking ? "checking…" : "not checked yet", reason: null };
    return { key, label, state: v ? "yes" : "no", value: v ? "available" : "unavailable", reason: v ? null : (describe(reasonIssue(issues, key)) ?? "The experiment's trainer could not be instantiated") };
  };
  return [
    { key: "metrics", label: "Metrics", state: c.metrics ? "yes" : "no", value: c.metrics ? "available" : "unavailable", reason: c.metrics ? null : describe(reasonIssue(issues, "metrics")) },
    {
      key: "params",
      label: "Parameters",
      state: c.params === "full" ? "yes" : c.params === "none" ? "no" : "partial",
      value: c.params,
      reason: c.params === "full" ? null : describe(reasonIssue(issues, "params")),
    },
    bool("replay", "Replay"),
    bool("launch", "Launch"),
  ];
}

/**
 * Tooltip of a disabled "Start runs" button, e.g. "Cannot start runs: the experiment's trainer
 * could not be deserialized (Unknown subclass … at trainer.mixer). See Issues."
 *
 * @ai-generated
 */
export function launchDisabledReason(c: Capabilities | null | undefined, issues: Issue[], checking = false): string | null {
  if (!c) return "Loading the experiment…";
  if (c.launch === true) return null;
  if (c.launch === null) return checking ? "Checking whether the experiment can be launched…" : "Launchability not checked yet";
  const i = reasonIssue(issues, "launch");
  const why = i ? `${i.message}${i.path && !i.message.includes(i.path) ? ` at ${i.path}` : ""}` : "unknown reason";
  return `Cannot start runs: the experiment's trainer could not be deserialized (${why}). See Issues.`;
}
