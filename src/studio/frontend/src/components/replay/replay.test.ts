import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { setApi, type Api } from "../../api";
import { createMockApi } from "../../api/mock";
import { groupBySeed, nAgents } from "../../domain/replay";
import { useReplayStore } from "../../stores/replay";
import { TRACKS_KEY, parseTracks, useTracksStore } from "../../stores/tracks";
import { useWorkspaceStore } from "../../stores/workspace";

const tick = (ms = 5) => new Promise((r) => setTimeout(r, ms));
async function settle(): Promise<void> {
  for (let i = 0; i < 8; i++) await tick();
}

let api: Api;
beforeEach(() => {
  localStorage.clear();
  setActivePinia(createPinia());
  api = createMockApi({ latency: [0, 0], tickMs: 1e9 });
  setApi(api);
});

const VDN = "lle5x5-vdn-mem50k";
const MAVEN = "lle5x5-maven-legacy";

describe("replay store", () => {
  it("opens at the last test step; timeline metric defaults per rule and is remembered per experiment", async () => {
    const r = useReplayStore();
    const ws = useWorkspaceStore();
    r.openEpisodes(VDN, null);
    await settle();
    const steps = await api.getTestSteps(VDN);
    expect(r.open).toBe(true);
    expect(r.step).toBe(steps[steps.length - 1]);
    expect(r.metric).toEqual({ table: "test", metric: "score-0" });

    r.setMetric({ table: "test", metric: "exit_rate" });
    expect(ws.ws.replayMetric[VDN]).toBe("test/exit_rate");
    r.openEpisodes("lle5x5-vdn-mem200k", 0);
    await settle();
    expect(r.metric?.metric).toBe("score-0");
    r.openEpisodes(VDN, null);
    await settle();
    expect(r.metric?.metric).toBe("exit_rate");
  });

  it("snaps steps to test steps, steps with arrows, and groups episodes by seed", async () => {
    const r = useReplayStore();
    r.openEpisodes(VDN, 20_400);
    await settle();
    const steps = r.steps;
    expect(r.step).toBe(20_000);
    expect(steps).toContain(20_000);
    r.setStep(steps[5] + 1);
    expect(r.step).toBe(steps[5]);
    r.stepBy(1);
    expect(r.step).toBe(steps[6]);
    r.stepBy(-100);
    expect(r.step).toBe(steps[0]);
    r.setStep(steps[5]);
    await settle();
    expect(r.episodes.length).toBeGreaterThan(0);
    const groups = groupBySeed(r.episodes);
    expect(groups.map((g) => g.seed)).toEqual([0, 1, 2, 3, 4]);
    expect(groups.every((g) => g.episodes.length === 4)).toBe(true);
  });

  it("loads the replay of a selected episode (capabilities checked lazily)", async () => {
    const r = useReplayStore();
    r.openEpisodes(VDN, null);
    await settle();
    expect(r.canReplay).toBe(true);
    await r.selectEpisode(r.episodes[0]);
    expect(r.replayStatus).toBe("ok");
    expect(r.replay && nAgents(r.replay)).toBe(2);
    expect(r.replay?.agent_details[0].q_values).toBeTruthy();
    r.seek(1000);
    expect(r.t).toBe(r.maxT);
    r.setStep(r.steps[1]);
    expect(r.selected).toBeNull();
    expect(r.replay).toBeNull();
  });

  it("gates the replay when capabilities.replay is false and keeps the episode metrics", async () => {
    const spy = vi.spyOn(api, "getReplay");
    const r = useReplayStore();
    r.openEpisodes(MAVEN, null);
    await settle();
    expect(r.canReplay).toBe(false);
    expect(r.blockingIssue?.message).toBeTruthy();
    const e = r.episodes[0] ?? { run: `${MAVEN}/run-0`, seed: 0, test: 0, step: r.step ?? 0, metrics: { "score-0": 1 }, has_actions: false };
    await r.selectEpisode(e);
    expect(spy).not.toHaveBeenCalled();
    expect(r.selected).toMatchObject({ run: e.run, test: e.test });
    expect(r.replay).toBeNull();
    expect(r.replayStatus).toBe("idle");
  });
});

describe("tracks store", () => {
  it("persists selections per experiment and survives a reload", () => {
    const t = useTracksStore();
    t.set("a", [
      { label: "Rewards", kind: "numeric" },
      { label: "q Agent 0", kind: "numeric" },
      { label: "Rewards", kind: "categorical" },
    ]);
    expect(t.forExperiment("a").map((x) => x.label)).toEqual(["Rewards", "q Agent 0"]);
    t.swap("a", 0, 1);
    t.update("a", { label: "Rewards", kind: "categorical" });
    t.add("b", { label: "options", kind: "categorical" });
    t.remove("b", "options");
    setActivePinia(createPinia());
    const again = useTracksStore();
    expect(again.forExperiment("a")).toEqual([
      { label: "q Agent 0", kind: "numeric" },
      { label: "Rewards", kind: "categorical" },
    ]);
    expect(again.forExperiment("b")).toEqual([]);
    expect(JSON.parse(localStorage.getItem(TRACKS_KEY)!)).not.toHaveProperty("b");
  });

  it("drops unreadable entries one by one", () => {
    expect(parseTracks({ a: [{ label: "x", kind: "weird" }, 3, { kind: "numeric" }], b: "nope" })).toEqual({ a: [{ label: "x", kind: "numeric" }] });
    expect(parseTracks("garbage")).toEqual({});
  });
});
