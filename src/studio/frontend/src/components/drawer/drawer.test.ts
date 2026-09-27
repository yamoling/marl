import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { nextTick } from "vue";
import { setApi } from "../../api";
import { createMockApi } from "../../api/mock";
import { useRunActions } from "../../composables/useRunActions";
import { flatten } from "../../domain/params";
import { useConfirmStore } from "../../stores/confirm";
import { useExperimentsStore } from "../../stores/experiments";
import { useUiStore } from "../../stores/ui";
import { useWorkspaceStore } from "../../stores/workspace";
import ParamDiffOverlay from "../diff/ParamDiffOverlay.vue";
import RenameDialog from "../launch/RenameDialog.vue";
import StartRunsDialog from "../launch/StartRunsDialog.vue";
import LaunchButton from "../launch/LaunchButton.vue";
import ExperimentDrawer from "./ExperimentDrawer.vue";
import ParamTree from "./ParamTree.vue";

const VDN = "lle5x5-vdn-mem50k";
const MAVEN = "lle5x5-maven-legacy";
const QMIX = "lle5x5-qmix-embed64";
const wait = async () => {
  for (let i = 0; i < 6; i++) {
    await new Promise((r) => setTimeout(r, 5));
    await flushPromises();
  }
};
const $ = <T extends Element = HTMLElement>(sel: string) => document.body.querySelector<T>(sel);
const $$ = (sel: string) => [...document.body.querySelectorAll<HTMLElement>(sel)];

beforeEach(() => {
  localStorage.clear();
  setActivePinia(createPinia());
  setApi(createMockApi({ latency: [0, 0], tickMs: 1e9 }));
});
afterEach(() => {
  document.body.innerHTML = "";
});

describe("ParamTree", () => {
  it("search highlights matches and auto-expands their ancestors", async () => {
    const rows = flatten({ trainer: { mixer: { embed_size: 64, "class-name": "QMix" }, lr: 1 } });
    const w = mount(ParamTree, { props: { rows } });
    expect(w.findAll(".row").map((r) => r.attributes("data-path"))).toEqual(["trainer", "trainer.mixer", "trainer.lr"]);
    await w.setProps({ query: "embed" });
    expect(w.findAll(".row").map((r) => r.attributes("data-path"))).toEqual(["trainer", "trainer.mixer", "trainer.mixer.embed_size"]);
    expect(w.find('.row[data-path="trainer.mixer.embed_size"]').classes()).toContain("match");
    expect(w.find("mark").text()).toBe("embed");
    expect(w.find('.row[data-path="trainer.mixer"]').attributes("aria-expanded")).toBe("true");
  });
});

describe("ExperimentDrawer", () => {
  it("opens on an experiment, runs the lazy health check and switches tabs", async () => {
    const ex = useExperimentsStore();
    ex.load(VDN);
    await wait();
    expect(ex.detail(VDN)!.capabilities.launch).toBeNull();
    const ui = useUiStore();
    const w = mount(ExperimentDrawer, { attachTo: document.body });
    ui.openDrawer(VDN);
    await wait();
    expect(ex.detail(VDN)!.capabilities.launch).toBe(true);
    expect($("h2")!.textContent).toBe(VDN);
    expect($$(".spec dt").map((d) => d.textContent)).toContain("Memory size");
    $<HTMLButtonElement>('[data-tab="params"]')!.click();
    await nextTick();
    const search = $<HTMLInputElement>('input[aria-label="Search parameters"]')!;
    search.value = "end_value";
    search.dispatchEvent(new Event("input"));
    await nextTick();
    expect($$(".tree .row").map((r) => r.dataset.path)).toEqual(["trainer", "trainer.train_policy", "trainer.train_policy.epsilon", "trainer.train_policy.epsilon.end_value"]);
    $<HTMLButtonElement>('[data-tab="issues"]')!.click();
    await nextTick();
    expect($$(".caps li").map((l) => l.dataset.cap)).toEqual(["metrics", "params", "replay", "launch"]);
    expect($('.caps li[data-cap="launch"]')!.classList).toContain("yes");
    w.unmount();
  });
});

describe("LaunchButton and StartRunsDialog", () => {
  it("disabled with the blocking issue for a non-launchable experiment", async () => {
    const ex = useExperimentsStore();
    ex.load(MAVEN);
    await wait();
    const w = mount(LaunchButton, { props: { id: MAVEN } });
    await wait();
    expect(w.find("button").attributes("disabled")).toBeDefined();
    expect(w.find("span").attributes("data-tip")).toMatch(/^Cannot start runs: .*QMixerV1.*See Issues\.$/);
  });

  it("validates, previews, reports seed collisions inline and starts runs", async () => {
    const ex = useExperimentsStore();
    ex.load(VDN);
    await wait();
    const ui = useUiStore();
    const w = mount(StartRunsDialog, { attachTo: document.body });
    ui.launchFor = VDN;
    await wait();
    expect($('[data-role="preview"]')!.textContent).toBe(`Will create run-5 in ${VDN}`);
    const setField = async (k: string, v: string) => {
      const i = $<HTMLInputElement>(`input[data-field="${k}"]`)!;
      i.value = v;
      i.dispatchEvent(new Event("input"));
      await nextTick();
    };
    await setField("seed", "3");
    expect($(".fe[role=alert]")!.textContent).toBe("Seed 3 already exists (next free: 5)");
    const submit = $<HTMLButtonElement>('button[type="submit"]')!;
    expect(submit.disabled).toBe(true);
    await setField("n_runs", "0");
    expect($$(".fe").map((e) => e.textContent)).toContain("Number of runs must be an integer ≥ 1");
    await setField("n_runs", "2");
    await setField("seed", "5");
    expect($('[data-role="preview"]')!.textContent).toBe(`Will create run-5, run-6 in ${VDN}`);
    expect(submit.disabled).toBe(false);
    submit.click();
    await wait();
    expect(ui.launchFor).toBeNull();
    expect(ex.detail(VDN)!.runs.map((r) => r.dirname)).toContain("run-6");
    w.unmount();
  });

  it("shows server errors inline (seed collision raced by another launch)", async () => {
    const ex = useExperimentsStore();
    ex.load(VDN);
    await wait();
    const ui = useUiStore();
    const w = mount(StartRunsDialog, { attachTo: document.body });
    ui.launchFor = VDN;
    await wait();
    await ex.startRuns(VDN, { n_runs: 1, seed: 5, n_tests: 1, test_interval: 1, n_jobs: 1, device: "auto", gpu_strategy: "group", disabled_devices: [], save_weights: false, save_actions: true });
    $<HTMLButtonElement>('button[type="submit"]')!.click();
    await wait();
    expect(ui.launchFor).toBe(VDN);
    expect($(".fe[role=alert]")!.textContent).toMatch(/Seed/);
    w.unmount();
  });
});

describe("Parameter diff overlay", () => {
  it("only differences by default, with the count; row → colour all plots", async () => {
    const ex = useExperimentsStore();
    ex.load([VDN, QMIX]);
    await wait();
    const ws = useWorkspaceStore();
    ws.createPlot({ title: "A" });
    const ui = useUiStore();
    const w = mount(ParamDiffOverlay, { attachTo: document.body });
    ui.diffOpen = true;
    await nextTick();
    const onlyDiff = $<HTMLInputElement>('[data-act="only-diff"]')!;
    expect(onlyDiff.checked).toBe(true);
    const count = $('[data-role="count"]')!.textContent!;
    const [, differing, total] = count.match(/(\d+) of (\d+) parameters differ/)!.map(Number);
    expect(differing).toBeGreaterThan(0);
    expect(total).toBeGreaterThan(differing);
    expect($$("tbody tr")).toHaveLength(differing);
    onlyDiff.click();
    await nextTick();
    expect($$("tbody tr")).toHaveLength(total);
    $<HTMLElement>('tr[data-path="trainer.memory_size"]')?.click();
    $<HTMLElement>('tr[data-path="trainer.mixer"]')!.click();
    await nextTick();
    $<HTMLButtonElement>('[data-act="colour-all"]')!.click();
    expect(ws.plots[0].colourBy).toEqual({ kind: "param", path: "trainer.mixer" });
    w.unmount();
  });
});

describe("rename and delete flows", () => {
  it("rename: client validation, 409 inline, then success rewrites the workspace", async () => {
    const ex = useExperimentsStore();
    ex.load([VDN, QMIX]);
    await wait();
    const ws = useWorkspaceStore();
    const p = ws.createPlot({ experiments: [VDN], hidden: [`${VDN}|test/score-0`] });
    const ui = useUiStore();
    ui.openDrawer(VDN);
    const w = mount(RenameDialog, { attachTo: document.body });
    useRunActions().rename(VDN);
    await nextTick();
    const input = $<HTMLInputElement>('input[aria-label="New experiment id"]')!;
    const submit = $<HTMLButtonElement>('button[type="submit"]')!;
    input.value = "bad name";
    input.dispatchEvent(new Event("input"));
    await nextTick();
    expect(submit.disabled).toBe(true);
    input.value = "lle5x5-ippo-clip0.2";
    input.dispatchEvent(new Event("input"));
    await nextTick();
    submit.click();
    await wait();
    expect($(".err")!.textContent).toMatch(/already exists/);
    input.value = "renamed/vdn";
    input.dispatchEvent(new Event("input"));
    await nextTick();
    submit.click();
    await wait();
    expect(ui.renameFor).toBeNull();
    expect(ui.drawerId).toBe("renamed/vdn");
    expect(ex.loaded).toEqual(["renamed/vdn", QMIX]);
    expect(ws.plot(p.id)).toMatchObject({ experiments: ["renamed/vdn"], hidden: ["renamed/vdn|test/score-0"] });
    expect(ex.entry("renamed/vdn")?.status).toBe("ready");
    w.unmount();
  });

  it("delete: typed confirmation, refused while running, then removed everywhere", async () => {
    const ex = useExperimentsStore();
    const LIVE = "lle5x5-qmix-embed128-live";
    ex.load([VDN, LIVE]);
    await wait();
    const ws = useWorkspaceStore();
    const p = ws.createPlot({ experiments: [VDN, LIVE] });
    const confirm = useConfirmStore();
    const actions = useRunActions();

    const refused = actions.remove(LIVE);
    await nextTick();
    expect(confirm.current?.confirmText).toBe(LIVE);
    await confirm.confirm();
    expect(confirm.error).toMatch(/still running/);
    confirm.cancel();
    expect(await refused).toBeUndefined();
    expect(ex.loaded).toContain(LIVE);

    const done = actions.remove(VDN);
    await nextTick();
    await confirm.confirm();
    await done;
    expect(confirm.current).toBeNull();
    expect(ex.loaded).toEqual([LIVE]);
    expect(ws.plot(p.id)!.experiments).toEqual([LIVE]);
    expect(ex.entry(VDN)).toBeUndefined();
  });
});
