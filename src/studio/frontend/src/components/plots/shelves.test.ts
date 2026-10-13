import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { nextTick } from "vue";
import { setApi } from "../../api";
import { createMockApi } from "../../api/mock";
import { endDrag } from "../../composables/dnd";
import type { ParamField } from "../../domain/fields";
import type { YField } from "../../domain/plot";
import { useExperimentsStore } from "../../stores/experiments";
import FieldsPanel from "../fields/FieldsPanel.vue";
import ColourByShelf from "./ColourByShelf.vue";
import RunsShelf from "./RunsShelf.vue";
import YShelf from "./YShelf.vue";

const dt = (payload: unknown) => ({ getData: () => JSON.stringify(payload), dropEffect: "" });
const Y: YField[] = [
  { table: "train", metric: "score-0", axis: "left" },
  { table: "test", metric: "score-0", axis: "left" },
];

afterEach(() => {
  document.body.innerHTML = "";
  endDrag();
});

describe("YShelf", () => {
  it("toggles a pill's axis and removes pills", async () => {
    const w = mount(YShelf, { props: { y: Y, groups: [], n: 0 } });
    expect(w.findAll("button.axis").map((b) => b.text())).toEqual(["L", "L"]);
    await w.findAll("button.axis")[1].trigger("click");
    expect(w.emitted("update:y")![0][0]).toEqual([Y[0], { ...Y[1], axis: "right" }]);
    await w.setProps({ y: [Y[0], { ...Y[1], axis: "right" }] });
    expect(w.findAll("button.axis").map((b) => b.text())).toEqual(["L", "R"]);
    await w.findAll("button.rm")[0].trigger("click");
    expect(w.emitted("update:y")![1][0]).toEqual([{ ...Y[1], axis: "right" }]);
  });

  it("accepts dropped metrics (no duplicates) and ignores parameters", async () => {
    const w = mount(YShelf, { props: { y: Y, groups: [], n: 0 } });
    const body = w.find(".shelf-body");
    await body.trigger("drop", { dataTransfer: dt({ kind: "metric", table: "training_data", metric: "td-loss" }) });
    expect(w.emitted("update:y")![0][0]).toEqual([...Y, { table: "training_data", metric: "td-loss", axis: "left" }]);
    await body.trigger("drop", { dataTransfer: dt({ kind: "metric", table: "test", metric: "score-0" }) });
    await body.trigger("drop", { dataTransfer: dt({ kind: "param", path: "trainer.lr" }) });
    expect(w.emitted("update:y")).toHaveLength(1);
  });

  it("adds a metric from the + menu", async () => {
    const groups = [{ table: "test", label: "Test", dash: "", colour: "#000", metrics: [{ table: "test", metric: "exit_rate", ids: ["a"], missing: [] }] }];
    const w = mount(YShelf, { props: { y: [], groups, n: 1 }, attachTo: document.body });
    await w.find("button.addbtn").trigger("click");
    const item = document.body.querySelector<HTMLButtonElement>(".pop .mi")!;
    expect(item.textContent).toContain("exit_rate");
    item.click();
    expect(w.emitted("update:y")![0][0]).toEqual([{ table: "test", metric: "exit_rate", axis: "left" }]);
    w.unmount();
  });
});

describe("ColourByShelf", () => {
  const params: ParamField[] = [
    { path: "trainer.memory_size", key: "memory_size", prefix: "trainer.", kind: "number", depth: 1, values: ["50000", "200000"], ids: ["a", "b"], distinct: 2, differs: true },
    { path: "trainer.lr", key: "lr", prefix: "trainer.", kind: "number", depth: 1, values: ["1", "1"], ids: ["a", "b"], distinct: 1, differs: false },
  ];

  it("selects a parameter from the picker and clears it", async () => {
    const w = mount(ColourByShelf, { props: { colourBy: { kind: "experiment" }, params, runsMode: "aggregate" }, attachTo: document.body });
    await w.find('[data-act="pick-param"]').trigger("click");
    document.body.querySelector<HTMLButtonElement>('.pop .mi[data-path="trainer.memory_size"]')!.click();
    expect(w.emitted("update:colourBy")![0][0]).toEqual({ kind: "param", path: "trainer.memory_size" });
    await w.setProps({ colourBy: { kind: "param", path: "trainer.memory_size" } });
    expect(w.find("button.on").text()).toBe("◆ memory_size");
    await w.find('[aria-label="Stop colouring by parameter"]').trigger("click");
    expect(w.emitted("update:colourBy")![1][0]).toEqual({ kind: "experiment" });
    w.unmount();
  });

  it("switches kinds and accepts dropped parameters", async () => {
    const w = mount(ColourByShelf, { props: { colourBy: { kind: "experiment" }, params, runsMode: "aggregate" } });
    const seed = w.findAll("button[role=radio]").find((b) => b.text() === "seed")!;
    expect(seed.attributes("title")).toMatch(/falls back/);
    await seed.trigger("click");
    expect(w.emitted("update:colourBy")![0][0]).toEqual({ kind: "seed" });
    await w.find(".shelf-body").trigger("drop", { dataTransfer: dt({ kind: "param", path: "trainer.lr" }) });
    expect(w.emitted("update:colourBy")![1][0]).toEqual({ kind: "param", path: "trainer.lr" });
  });
});

describe("RunsShelf", () => {
  const exps = [
    { id: "a", name: "a", colour: "#111", runs: [0, 1, 2].map((s) => ({ id: `a/run-${s}`, seed: s })) },
    { id: "b", name: "b", colour: "#222", runs: [0, 1].map((s) => ({ id: `b/run-${s}`, seed: s })) },
  ];

  it("switches modes", async () => {
    const w = mount(RunsShelf, { props: { runs: { mode: "aggregate", seeds: null }, experiments: exps } });
    const radios = w.findAll("button[role=radio]");
    expect(radios.map((r) => r.text())).toEqual(["Aggregate", "Aggregate + runs", "Runs only"]);
    await radios[2].trigger("click");
    await radios[1].trigger("click");
    expect(w.emitted("update:runs")!.map((e) => e[0])).toEqual([
      { mode: "runs", seeds: null },
      { mode: "aggregate+runs", seeds: null },
    ]);
  });

  it("seed picker builds per-experiment subsets and back to all", async () => {
    const w = mount(RunsShelf, { props: { runs: { mode: "runs", seeds: null }, experiments: exps }, attachTo: document.body });
    expect(w.find('[data-act="seeds"]').text()).toContain("all seeds");
    await w.find('[data-act="seeds"]').trigger("click");
    const box = document.body.querySelector<HTMLInputElement>('input[data-seed="a:1"]')!;
    box.checked = false;
    box.dispatchEvent(new Event("change"));
    expect(w.emitted("update:runs")![0][0]).toEqual({ mode: "runs", seeds: { a: [0, 2] } });
    await w.setProps({ runs: { mode: "runs", seeds: { a: [0, 2] } } });
    expect(w.find('[data-act="seeds"]').text()).toContain("4/5 runs");
    const back = document.body.querySelector<HTMLInputElement>('input[data-seed="a:1"]')!;
    back.checked = true;
    back.dispatchEvent(new Event("change"));
    expect(w.emitted("update:runs")![1][0]).toEqual({ mode: "runs", seeds: null });
    w.unmount();
  });
});

describe("FieldsPanel", () => {
  beforeEach(() => {
    localStorage.clear();
    setActivePinia(createPinia());
    setApi(createMockApi({ latency: [0, 0], tickMs: 1e9 }));
  });

  it("shows k/n counters with the missing experiments in the tooltip, and differing parameters first", async () => {
    const ex = useExperimentsStore();
    ex.load(["lle5x5-vdn-mem50k", "lle5x5-ippo-clip0.2", "lle5x5-qmix-embed64"]);
    for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 5));
    const w = mount(FieldsPanel);
    await nextTick();
    const groups = w.findAll(".grp").map((g) => g.attributes("data-table"));
    expect(groups).toEqual(["test", "train", "training_data", "test-policy-on-test-envs"]);
    const pill = (table: string, metric: string) => w.findAll(`.grp[data-table="${table}"] .fpill`).find((p) => p.text().startsWith(metric))!;
    expect(pill("test", "score-0").find(".cnt").text()).toBe("3/3");
    expect(pill("test", "score-0").find(".cnt").classes()).not.toContain("partial");
    const td = pill("training_data", "td-loss");
    expect(td.find(".cnt").text()).toBe("2/3");
    expect(td.find(".cnt").classes()).toContain("partial");
    expect(td.attributes("data-tip")).toBe("training_data/td-loss is missing in: ippo-clip0.2");
    expect(pill("test-policy-on-test-envs", "score-0").find(".cnt").text()).toBe("1/3");

    const params = w.findAll(".fpill.param");
    expect(params[0].find(".pv").classes()).toContain("diff");
    expect(params.some((p) => p.text().includes("memory_size"))).toBe(true);

    await w.find('input[aria-label="Filter fields"]').setValue("td-l");
    expect(w.findAll(".grp").map((g) => g.attributes("data-table"))).toEqual(["training_data"]);
    w.unmount();
  });
});
