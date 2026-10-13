import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";
import { setApi } from "../../api";
import { createMockApi } from "../../api/mock";
import { MIME } from "../../composables/dnd";
import { usePlotActions } from "../../composables/usePlotActions";
import { PRESETS } from "../../domain/plot";
import { useExperimentsStore } from "../../stores/experiments";
import { useWorkspaceStore } from "../../stores/workspace";
import PlotGrid from "./PlotGrid.vue";
import PresetsPanel from "./PresetsPanel.vue";

const EXPERIMENT = "lle5x5-vdn-mem50k";
let wrapper: VueWrapper | undefined;

beforeEach(() => {
    localStorage.clear();
    setActivePinia(createPinia());
    setApi(createMockApi({ latency: [0, 0], tickMs: 1e9 }));
    // jsdom has no layout; give the real SVG renderer a usable viewport.
    vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockReturnValue(600);
    vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockReturnValue(300);
});
afterEach(() => {
    wrapper?.unmount();
    wrapper = undefined;
    document.body.innerHTML = "";
    vi.restoreAllMocks();
});

/** Load the real mock catalog, then mount the grid with a train/test comparison. @ai-generated */
async function comparison() {
    const experiments = useExperimentsStore();
    experiments.load(EXPERIMENT);
    await vi.waitFor(() => expect(experiments.catalog(EXPERIMENT)).not.toBeNull());
    const workspace = useWorkspaceStore();
    const preset = PRESETS.trainVsTest(experiments.context);
    expect(preset).not.toBeNull();
    const plot = workspace.createPlot(preset!);
    wrapper = mount(PlotGrid, { attachTo: document.body });
    await vi.waitFor(() => expect(wrapper!.findAll("path.mc-center")).toHaveLength(2));
    return { workspace, plot };
}

/** Find a button by its accessible label or complete visible text. @ai-generated */
function button(text: string) {
    const found = wrapper!.findAll("button").find((b) => (b.attributes("aria-label") ?? b.text()) === text);
    if (!found) throw new Error(`Button not found: ${text}`);
    return found;
}

describe("PresetsPanel", () => {
    it("offers applicable presets after catalogs load and creates the selected plot", async () => {
        wrapper = mount(PresetsPanel);
        expect(wrapper.findAll("button")).toHaveLength(0);
        const experiments = useExperimentsStore();
        experiments.load(EXPERIMENT);
        await vi.waitFor(() => expect(experiments.catalog(EXPERIMENT)).not.toBeNull());
        await nextTick();
        await button("Train vs test").trigger("click");
        const workspace = useWorkspaceStore();
        expect(workspace.plots).toHaveLength(1);
        expect(workspace.plots[0].title).toBe("Train vs test");
        expect(workspace.plots[0].y.map((field) => field.table)).toEqual(["train", "test"]);
    });
});

describe("PlotGrid", () => {
    it("renders compact icon actions with accessible names and tooltips", async () => {
        await comparison();
        for (const name of ["Shelves", "Minimize", "Maximize", "Delete plot", "Reset zoom", "Duplicate"]) {
            const action = button(name);
            expect(action.text()).toBe("");
            expect(action.attributes("title")).toBeTruthy();
            expect(action.find('svg[aria-hidden="true"]').exists()).toBe(true);
        }
    });
    it("renders train and test from the shared cache and toggles legend visibility", async () => {
        const { plot } = await comparison();
        expect(wrapper!.find("article").attributes("aria-label")).toBe("Plot Train vs test");
        expect(wrapper!.text()).toContain("2 series");
        for (const path of wrapper!.findAll("path.mc-center")) expect(path.attributes("d")).toMatch(/^M/);
        const legend = wrapper!.find(".legend button");
        await legend.trigger("click");
        expect(plot.hidden).toHaveLength(1);
        expect(legend.attributes("aria-pressed")).toBe("false");
        await legend.trigger("click");
        expect(plot.hidden).toEqual([]);
    });

    it("maximizes one card without rendering a duplicate, restores, and expands a minimized card", async () => {
        const { workspace, plot } = await comparison();
        await button("Maximize").trigger("click");
        expect(workspace.maximizedId).toBe(plot.id);
        await nextTick();
        const dialog = document.querySelector('[role="dialog"][aria-label="Train vs test"]')!;
        expect(dialog).not.toBeNull();
        expect(document.querySelectorAll('article[aria-label="Plot Train vs test"]')).toHaveLength(1);
        const restore = [...dialog.querySelectorAll("button")].find((b) => b.getAttribute("aria-label") === "Restore")!;
        restore.click();
        await nextTick();
        expect(workspace.maximizedId).toBeNull();
        await button("Minimize").trigger("click");
        expect(plot.view).toBe("minimized");
        expect(wrapper!.find(".line-chart").exists()).toBe(false);
        await button("Expand").trigger("click");
        expect(plot.view).toBe("normal");
        expect(wrapper!.find(".line-chart").exists()).toBe(true);
    });

    it("opens metric selection for an empty plot and deduplicates selected fields", async () => {
        await comparison();
        const actions = usePlotActions();
        const plot = actions.newEmptyPlot();
        await nextTick();
        await flushPromises();
        const menu = document.querySelector('[role="dialog"][aria-label="Add metric"]')!;
        expect(menu).not.toBeNull();
        const metric = [...menu.querySelectorAll("button")].find((b) => b.textContent?.includes("score-0"))!;
        expect(metric).toBeDefined();
        metric.click();
        await nextTick();
        expect(plot.y).toHaveLength(1);
        actions.applyField(plot.id, { kind: "metric", table: plot.y[0].table, metric: plot.y[0].metric });
        expect(plot.y).toHaveLength(1);
        expect(wrapper!.findAll("article")).toHaveLength(2);
    });

    it("applies dropped fields and creates a new plot from a dropped metric", async () => {
        const { plot, workspace } = await comparison();
        const payload = { kind: "metric", table: "test", metric: "exit_rate" };
        const dataTransfer = { getData: (type: string) => (type === MIME ? JSON.stringify(payload) : "") };
        await wrapper!.find("article").trigger("drop", { dataTransfer });
        expect(plot.y).toContainEqual({ table: "test", metric: "exit_rate", axis: "left" });
        await wrapper!.find(".new-plot").trigger("drop", { dataTransfer });
        expect(workspace.plots).toHaveLength(2);
        expect(workspace.plots[1].title).toBe("exit_rate (test)");
    });

    it("updates shelves, duplicates plots, and deletes with the store's undo entry", async () => {
        const { workspace, plot } = await comparison();
        await button("Shelves").trigger("click");
        await wrapper!.find('select[aria-label="X axis"]').setValue("wall_time");
        await wrapper!.find('select[aria-label="Band"]').setValue("none");
        expect(plot.x.axis).toBe("wall_time");
        expect(plot.stat.band).toBe("none");
        await button("Duplicate").trigger("click");
        expect(workspace.plots).toHaveLength(2);
        expect(workspace.plots[1].title).toBe("Train vs test (copy)");
        await button("Delete plot").trigger("click");
        expect(workspace.plots).toHaveLength(1);
        workspace.undo();
        expect(workspace.plots).toHaveLength(2);
    });
});
