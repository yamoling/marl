import { describe, expect, it } from "vitest";
import { diff, flatten, paramValue, scheduleValue } from "./params";
import { defaultMetric, lossMetrics, sortTables } from "./metrics";
import { fmtRelativeDate, fmtStep, fmtTick } from "./format";

const RAW = {
  n_steps: 1000,
  trainer: {
    lr: 5e-4,
    mixer: { embed_size: 64, "class-name": "QMix", name: "QMix" },
    train_policy: {
      epsilon: { start_value: 1, end_value: 0.05, n_steps: 100, "class-name": "LinearSchedule", name: "LinearSchedule" },
      "class-name": "EpsilonGreedy",
      name: "custom",
    },
    mlp_sizes: [64, 64],
    ir: null,
    "class-name": "DQN",
  },
};

describe("flatten", () => {
  const rows = flatten(RAW);
  it("depth-first rows with cls on object nodes and hidden duplicate names", () => {
    expect(rows.map((r) => r.path)).toEqual([
      "n_steps",
      "trainer",
      "trainer.lr",
      "trainer.mixer",
      "trainer.mixer.embed_size",
      "trainer.train_policy",
      "trainer.train_policy.epsilon",
      "trainer.train_policy.epsilon.start_value",
      "trainer.train_policy.epsilon.end_value",
      "trainer.train_policy.epsilon.n_steps",
      "trainer.train_policy.name",
      "trainer.mlp_sizes",
      "trainer.ir",
    ]);
    expect(rows.find((r) => r.path === "trainer.mixer")).toMatchObject({ kind: "object", cls: "QMix", depth: 1, value: null });
    expect(rows.find((r) => r.path === "trainer.mlp_sizes")).toMatchObject({ kind: "array", value: [64, 64] });
    expect(rows.find((r) => r.path === "trainer.ir")).toMatchObject({ kind: "null" });
  });

  it("schedules have a curve mirroring LinearSchedule", () => {
    const eps = rows.find((r) => r.path === "trainer.train_policy.epsilon")!;
    expect(eps.kind).toBe("schedule");
    expect(eps.curve!.y[0]).toBe(1);
    expect(eps.curve!.y.at(-1)).toBe(0.05);
    expect(scheduleValue({ "class-name": "ExpSchedule", start_value: 1, end_value: 0.01, n_steps: 101 }, 50)).toBeCloseTo(0.1);
    expect(
      scheduleValue(
        {
          "class-name": "RoundedSchedule",
          n_digits: 0,
          schedule: { "class-name": "LinearSchedule", start_value: 0, end_value: 10, n_steps: 10 },
        },
        3.4,
      ),
    ).toBe(3);
    expect(scheduleValue({ "class-name": "WeirdSchedule" }, 1)).toBeNull();
  });

  it("paramValue and diff", () => {
    expect(paramValue(rows, "trainer.mixer")).toBe("QMix");
    expect(paramValue(rows, "trainer.lr")).toBe(5e-4);
    expect(paramValue(rows, "nope")).toBeUndefined();
    const other = flatten({ n_steps: 1000, trainer: { lr: 1e-3, "class-name": "DQN" } });
    const d = diff([rows, other]);
    expect(d.find((r) => r.path === "n_steps")!.differs).toBe(false);
    expect(d.find((r) => r.path === "trainer.lr")).toMatchObject({ differs: true, values: ["5.0e-4", "0.001"] });
    expect(d.find((r) => r.path === "trainer.mixer")!.values).toEqual(["QMix", "∅"]);
  });
});

describe("metrics rules and formatting", () => {
  const cat = (metrics: string[]) => ({ tables: { test: { metrics, x_columns: [], runs: [] } }, default_metric: null, loss_metrics: [] });
  it("default metric: score > score-0 > first alphabetical", () => {
    expect(defaultMetric(cat(["a", "score-0", "score"]))).toEqual({ table: "test", metric: "score" });
    expect(defaultMetric(cat(["z", "score-0"]))).toEqual({ table: "test", metric: "score-0" });
    expect(defaultMetric(cat(["time_step", "zeta", "alpha"]))).toEqual({ table: "test", metric: "alpha" });
    expect(defaultMetric({ ...cat([]), default_metric: { table: "t", metric: "m" } })).toEqual({ table: "t", metric: "m" });
    expect(
      lossMetrics({
        tables: { training_data: { metrics: ["td-loss", "q", "grad-norm"], x_columns: [], runs: [] } },
        default_metric: null,
        loss_metrics: [],
      }).map((m) => m.metric),
    ).toEqual(["td-loss", "grad-norm"]);
    expect(sortTables(["zz", "training_data", "test", "aa", "train"])).toEqual(["test", "train", "training_data", "aa", "zz"]);
  });
  it("formats", () => {
    expect([fmtTick(1_200_000), fmtTick(500_000), fmtTick(0.001), fmtTick(0.25)]).toEqual(["1.2M", "500k", "1e-3", "0.25"]);
    expect(fmtStep(1_234_567)).toBe("1.23M");
    const now = new Date("2026-09-26T12:00:00Z");
    expect(fmtRelativeDate("2026-09-26T11:55:00Z", now)).toBe("5 min ago");
    expect(fmtRelativeDate("2026-09-25T11:00:00Z", now)).toBe("yesterday");
    expect(fmtRelativeDate(null, now)).toBe("—");
  });
});
