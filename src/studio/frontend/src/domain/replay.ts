/**
 * Pure helpers over episodes and replays (port of the old UI's `models/Episode.ts` logic plus the
 * episode-card rules of the episodes sheet). Replay arrays are validated lazily here: every
 * accessor tolerates missing or malformed entries.
 */
import type { ActionSpace, EpisodeSummary, ReplayEpisode } from "../api/schemas";
import { track, type TrackNode } from "./timeline";

// ---------------------------------------------------------------- shapes (old `utils.ts`)

/** Shape of a nested array, following first elements. @ai-generated */
export function computeShape(value: unknown): number[] {
  const out: number[] = [];
  let a = value;
  while (Array.isArray(a)) {
    out.push(a.length);
    a = a[0];
  }
  return out;
}

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const numOrNull = (v: unknown): number | null => (isNum(v) ? v : typeof v === "boolean" ? Number(v) : null);

/** A vector of numbers (non-numbers become null), or null if `v` is not a flat array. @ai-generated */
export function asVector(v: unknown): (number | null)[] | null {
  if (!Array.isArray(v) || v.some((x) => Array.isArray(x))) return null;
  return v.map(numOrNull);
}

/** A matrix of numbers, or null if `v` is not an array of flat arrays. @ai-generated */
export function asMatrix(v: unknown): (number | null)[][] | null {
  if (!Array.isArray(v) || !v.length) return null;
  const rows = v.map(asVector);
  return rows.every((r): r is (number | null)[] => r !== null) ? rows : null;
}

// ---------------------------------------------------------------- frames, lengths, agents

/** `<img src>` of a frame: data URLs as-is, bare base64 as JPEG (the backend's encoding). */
export function frameSrc(frame: string | undefined): string {
  if (!frame) return "";
  return frame.startsWith("data:") ? frame : `data:image/jpeg;base64,${frame}`;
}

/** Number of transitions (actions) of the replayed episode. @ai-generated */
export function episodeLength(ep: ReplayEpisode): number {
  const n = ep.episode.episode_len;
  if (isNum(n) && n >= 0) return n;
  if (ep.episode.actions.length) return ep.episode.actions.length;
  return Math.max(0, ep.frames.length - 1);
}

/** Last displayable time index: frames go from the initial state (0) to the final one (length). */
export function maxTime(ep: ReplayEpisode): number {
  return Math.max(0, Math.max(episodeLength(ep), ep.frames.length - 1));
}

/** Number of agents, from the actions, the available actions or the action space. @ai-generated */
export function nAgents(ep: ReplayEpisode): number {
  const a0 = ep.episode.actions[0];
  if (Array.isArray(a0)) return a0.length;
  const av0 = ep.episode.all_available_actions[0];
  if (Array.isArray(av0)) return av0.length;
  return ep.action_space?.spaces?.length ?? 0;
}

export const isDiscreteSpace = (s: ActionSpace | null | undefined): boolean => Array.isArray(s?.spaces);

/** Action labels of agent `agent` (per-agent space labels, else the global labels). @ai-generated */
export function actionLabels(space: ActionSpace | null | undefined, agent = 0): string[] {
  if (!space) return [];
  const own = space.spaces?.[agent]?.labels ?? space.spaces?.[0]?.labels;
  return own?.length ? own : space.labels;
}

/** Number of discrete actions (size of the first sub-space, else the label count). @ai-generated */
export function nActions(space: ActionSpace | null | undefined): number {
  if (!space) return 0;
  return space.spaces?.[0]?.size || actionLabels(space).length;
}

// ---------------------------------------------------------------- per-step accessors

const at = (xs: readonly unknown[], t: number): unknown => (t >= 0 && t < xs.length ? xs[t] : undefined);
const agentAt = (xs: readonly unknown[], t: number, agent: number): unknown => {
  const row = at(xs, t);
  return Array.isArray(row) ? row[agent] : undefined;
};

/** Action of `agent` at time `t`: a discrete index, a continuous vector, or null (terminal step). @ai-generated */
export function actionAt(ep: ReplayEpisode, t: number, agent: number): number | number[] | null {
  const v = agentAt(ep.episode.actions, t, agent);
  if (isNum(v)) return v;
  const vec = asVector(v);
  return vec ? vec.map((x) => x ?? 0) : null;
}

/** Whether `action` is available to `agent` at `t` (unknown → available). @ai-generated */
export function isAvailableAt(ep: ReplayEpisode, t: number, agent: number, action: number): boolean {
  const row = agentAt(ep.episode.all_available_actions, t, agent);
  if (!Array.isArray(row) || action >= row.length) return true;
  const v = row[action];
  return typeof v === "boolean" ? v : typeof v === "number" ? v !== 0 : Boolean(v);
}

export const observationAt = (ep: ReplayEpisode, t: number, agent: number): unknown => agentAt(ep.episode.all_observations, t, agent);

export function extrasAt(ep: ReplayEpisode, t: number, agent: number): number[] {
  return (asVector(agentAt(ep.episode.all_extras, t, agent)) ?? []).map((x) => x ?? 0);
}

export const detailsAt = (ep: ReplayEpisode, t: number): Record<string, unknown> | null => ep.agent_details[t] ?? null;

export type DecisionKey = "q_values" | "action_probabilities";
export const DECISION_LABELS: Record<DecisionKey, string> = { q_values: "Q-values", action_probabilities: "Action probabilities" };

/** Which agent-detail key holds per-action values at `t` (q-values first). @ai-generated */
export function decisionKeyAt(ep: ReplayEpisode, t: number): DecisionKey | null {
  const d = detailsAt(ep, t);
  if (d?.q_values != null) return "q_values";
  if (d?.action_probabilities != null) return "action_probabilities";
  return null;
}

/**
 * Per-action values of one agent at `t` for `key`: a vector (one value per action) or a matrix
 * (action × objective, multi-objective q-values). Null when absent or malformed.
 *
 * @ai-generated
 */
export function decisionValues(
  ep: ReplayEpisode,
  t: number,
  key: DecisionKey,
  agent: number,
): (number | null)[] | (number | null)[][] | null {
  const raw = detailsAt(ep, t)?.[key];
  if (!Array.isArray(raw)) return null;
  const forAgent = raw[agent];
  return asVector(forAgent) ?? asMatrix(forAgent);
}

// ---------------------------------------------------------------- tracks (old `computeTracks`)

/**
 * Timeline tracks of a replay: rewards (one track per reward component), then every agent-detail
 * key: 3D values → a group of `key Agent i/j` tracks, 2D → a group of `key Agent i` tracks,
 * 1D (a scalar per step) → one track. Labels are identical to the old UI so stored selections carry over.
 *
 * @ai-generated
 */
export function computeTracks(ep: ReplayEpisode): TrackNode[] {
  const out: TrackNode[] = [];
  const rewards = ep.episode.rewards;
  if (rewards.length && Array.isArray(rewards[0])) {
    const n = (rewards[0] as unknown[]).length;
    // The old UI labelled every component "Rewards" (duplicates); components are numbered when there are several.
    for (let i = 0; i < n; i++)
      out.push(
        track(
          n > 1 ? `Rewards ${i}` : "Rewards",
          rewards.map((r) => numOrNull(Array.isArray(r) ? r[i] : null)),
        ),
      );
  } else if (rewards.length) {
    out.push(track("Rewards", rewards.map(numOrNull)));
  }

  const details = ep.agent_details;
  // Union of the keys of every step (the old UI only looked at the first step).
  const keys = [...new Set(details.flatMap((d) => Object.keys(d)))];
  const agents = nAgents(ep);
  for (const key of keys) {
    const values = details.map((d) => d[key]);
    // Shape from the first step that has the key (steps × …).
    const shape = [values.length, ...computeShape(values.find((v) => v != null))];
    if (shape.length === 3) {
      const subTracks = [];
      for (let i = 0; i < Math.min(agents || shape[1], shape[1]); i++)
        for (let j = 0; j < shape[2]; j++)
          subTracks.push(
            track(
              `${key} Agent ${i}/${j}`,
              values.map((v) => numOrNull((v as unknown[][] | undefined)?.[i]?.[j])),
            ),
          );
      out.push({ type: "group", label: key, subTracks });
    } else if (shape.length === 2) {
      const subTracks = [];
      for (let i = 0; i < Math.min(agents || shape[1], shape[1]); i++)
        subTracks.push(
          track(
            `${key} Agent ${i}`,
            values.map((v) => numOrNull((v as unknown[] | undefined)?.[i])),
          ),
        );
      out.push({ type: "group", label: key, subTracks });
    } else if (shape.length === 1) {
      out.push(track(key, values.map(numOrNull)));
    }
  }
  return out;
}

// ---------------------------------------------------------------- episode cards

export type Outcome = "success" | "partial" | "failure" | "unknown";

/** Success of a test episode from its `exit_rate` (1 → success, >0 → partial, 0 → failure). @ai-generated */
export function episodeOutcome(metrics: Record<string, unknown>): Outcome {
  const x = metrics.exit_rate;
  if (typeof x !== "number" || !Number.isFinite(x)) return "unknown";
  return x >= 1 ? "success" : x > 0 ? "partial" : "failure";
}

export type SeedGroup = { key: string; seed: number | null; run: string; episodes: EpisodeSummary[] };

/**
 * Episodes grouped by run (seed), sorted by seed (unknown seeds last, then run id), each group
 * sorted by test number.
 *
 * @ai-generated
 */
export function groupBySeed(episodes: readonly EpisodeSummary[]): SeedGroup[] {
  const groups = new Map<string, SeedGroup>();
  for (const e of episodes) {
    let g = groups.get(e.run);
    if (!g) groups.set(e.run, (g = { key: e.run, seed: e.seed, run: e.run, episodes: [] }));
    g.episodes.push(e);
  }
  const out = [...groups.values()];
  for (const g of out) g.episodes.sort((a, b) => a.test - b.test);
  return out.sort((a, b) => {
    if (a.seed === null || b.seed === null) return a.seed === b.seed ? a.run.localeCompare(b.run) : a.seed === null ? 1 : -1;
    return a.seed - b.seed || a.run.localeCompare(b.run);
  });
}

/** Key of the score shown on episode cards: the timeline metric if episodes have it, else `score`, else `score-0`. @ai-generated */
export function scoreKey(episodes: readonly EpisodeSummary[], preferred: string | null | undefined): string | null {
  const has = (k: string) => episodes.some((e) => k in e.metrics);
  for (const k of [preferred, "score", "score-0"]) if (k && has(k)) return k;
  return null;
}

const CARD_HIDDEN = new Set(["exit_rate", "episode_len", "time_step", "timestamp_sec"]);

/** Other scalar metrics shown as chips on a card (everything but score, exit and length). @ai-generated */
export function extraMetrics(metrics: Record<string, unknown>, score: string | null): [string, number | boolean | string][] {
  return Object.entries(metrics).filter(
    (e): e is [string, number | boolean | string] => e[0] !== score && !CARD_HIDDEN.has(e[0]) && e[1] !== null && typeof e[1] !== "object",
  );
}

/** Port of the old `numberFormat.ts`: integers as-is, other numbers with 3 decimals. @ai-generated */
export function formatNumber(value: number | string | boolean | null | undefined): string {
  if (value === null || value === undefined) return "–";
  if (typeof value === "boolean") return value ? "true" : "false";
  const n = typeof value === "string" ? Number.parseFloat(value) : value;
  if (Number.isNaN(n)) return String(value);
  if (!Number.isFinite(n)) return String(n);
  return Number.isInteger(n) ? String(n) : n.toFixed(3);
}

/** Issue codes that explain why an experiment cannot be replayed (most specific first). */
export const REPLAY_BLOCKING_CODES = [
  "deserialize-failed",
  "missing-keys",
  "missing-experiment-json",
  "replay-unavailable",
  "not-replayable",
];
