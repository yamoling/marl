import { describe, expect, it } from "vitest";
import { ReplayEpisodeSchema, type EpisodeSummary } from "../api/schemas";
import fixture from "../components/replay/__fixtures__/replay-lle.json";
import {
  actionAt,
  actionLabels,
  computeTracks,
  decisionKeyAt,
  decisionValues,
  episodeLength,
  episodeOutcome,
  extraMetrics,
  formatNumber,
  frameSrc,
  groupBySeed,
  isAvailableAt,
  isDiscreteSpace,
  maxTime,
  nActions,
  nAgents,
  scoreKey,
} from "./replay";

describe("ReplayEpisode schema (real LLE payload, trimmed)", () => {
  const ep = ReplayEpisodeSchema.parse(fixture);

  it("parses the real payload and its derived facts", () => {
    expect(ep.name).toBe("run-9");
    expect(ep.replay_kind).toBe("CombinedReplayAgent");
    expect(ep.replay_mismatch).toBe(false);
    expect(ep.time_step).toBe(10000);
    expect(ep.metrics["score-0"]).toBe(0);
    expect(nAgents(ep)).toBe(4);
    expect(episodeLength(ep)).toBe(4);
    expect(ep.frames).toHaveLength(5);
    expect(maxTime(ep)).toBe(4);
    expect(frameSrc(ep.frames[0])).toMatch(/^data:image\/jpeg;base64,\/9j\//);
    expect(isDiscreteSpace(ep.action_space)).toBe(true);
    expect(nActions(ep.action_space)).toBe(5);
    expect(actionLabels(ep.action_space, 2)).toEqual(["NORTH", "SOUTH", "EAST", "WEST", "STAY"]);
  });

  it("per-step accessors", () => {
    expect(actionAt(ep, 0, 0)).toBe(fixture.episode.actions[0][0]);
    expect(actionAt(ep, 99, 0)).toBeNull();
    expect(isAvailableAt(ep, 0, 0, 0)).toBe(fixture.episode.all_available_actions[0][0][0]);
    expect(decisionKeyAt(ep, 0)).toBe("q_values");
    const q = decisionValues(ep, 0, "q_values", 1);
    expect(q).toHaveLength(5);
    expect(q?.every((v) => typeof v === "number")).toBe(true);
    expect(decisionValues(ep, 0, "action_probabilities", 1)).toBeNull();
  });

  it("tracks: one reward component, q-values as a group of agent/action tracks (old UI labels)", () => {
    const tracks = computeTracks(ep);
    expect(tracks[0]).toMatchObject({ type: "track", label: "Rewards" });
    expect(tracks[0].type === "track" && tracks[0].values).toHaveLength(4);
    const q = tracks[1];
    expect(q.type).toBe("group");
    expect(q.type === "group" && q.subTracks.map((t) => t.label).slice(0, 2)).toEqual(["q_values Agent 0/0", "q_values Agent 0/1"]);
    expect(q.type === "group" && q.subTracks).toHaveLength(20);
  });

  it("tolerates a malformed payload (defaults instead of throwing)", () => {
    const bad = ReplayEpisodeSchema.parse({
      name: 3,
      episode: "nope",
      metrics: { a: 1, b: "x", c: true, d: null },
      frames: ["abc", 4],
      agent_details: [null, { k: 1 }],
      action_space: "?",
      replay_kind: "Other",
    });
    expect(bad.name).toBe("");
    expect(bad.metrics).toEqual({ a: 1, b: null, c: 1, d: null });
    expect(bad.frames).toEqual(["abc", ""]);
    expect(bad.agent_details).toEqual([{}, { k: 1 }]);
    expect(bad.action_space).toBeNull();
    expect(bad.replay_kind).toBe("UNKNOWN");
    expect(episodeLength(bad)).toBe(1);
    expect(nAgents(bad)).toBe(0);
    expect(computeTracks(bad)).toEqual([{ type: "track", label: "k", kind: "numeric", values: [null, 1] }]);
    expect(frameSrc("data:image/svg+xml;utf8,x")).toBe("data:image/svg+xml;utf8,x");
  });
});

const ep = (run: string, seed: number | null, test: number, metrics: EpisodeSummary["metrics"] = {}): EpisodeSummary => ({
  run,
  seed,
  test,
  step: 100,
  metrics,
  has_actions: true,
});

describe("episode cards", () => {
  it("groups episodes by seed (unknown seeds last) and sorts tests", () => {
    const groups = groupBySeed([ep("e/run-2", 2, 1), ep("e/x", null, 0), ep("e/run-10", 10, 0), ep("e/run-2", 2, 0), ep("e/run-1", 1, 3)]);
    expect(groups.map((g) => g.seed)).toEqual([1, 2, 10, null]);
    expect(groups[1].episodes.map((e) => e.test)).toEqual([0, 1]);
    expect(groups[1].run).toBe("e/run-2");
  });

  it("outcome from exit_rate, score key and extra chips", () => {
    expect(episodeOutcome({ exit_rate: 1 })).toBe("success");
    expect(episodeOutcome({ exit_rate: 0.5 })).toBe("partial");
    expect(episodeOutcome({ exit_rate: 0 })).toBe("failure");
    expect(episodeOutcome({})).toBe("unknown");
    const eps = [ep("a", 0, 0, { "score-0": 1, gems: 2 })];
    expect(scoreKey(eps, "gems")).toBe("gems");
    expect(scoreKey(eps, "missing")).toBe("score-0");
    expect(scoreKey([ep("a", 0, 0, { score: 1, "score-0": 2 })], null)).toBe("score");
    expect(extraMetrics({ "score-0": 1, exit_rate: 1, episode_len: 3, gems: 2, n: null }, "score-0")).toEqual([["gems", 2]]);
    expect(formatNumber(3)).toBe("3");
    expect(formatNumber(0.12345)).toBe("0.123");
    expect(formatNumber(null)).toBe("–");
  });
});
