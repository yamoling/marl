import { createPinia, setActivePinia } from "pinia";
import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it } from "vitest";
import { setApi } from "../api";
import { createMockApi } from "../api/mock";
import { router } from "../router";
import { useNamedWorkspacesStore } from "../stores/namedWorkspaces";
import { useWorkspaceStore } from "../stores/workspace";
import { emptyWorkspace, STORAGE_KEY } from "../domain/workspace";
import HomePage from "./HomePage.vue";

const settle = () => new Promise((resolve) => setTimeout(resolve, 20));

beforeEach(async () => {
  localStorage.clear();
  setActivePinia(createPinia());
  setApi(createMockApi({ latency: [0, 0], tickMs: 1e9 }));
  await router.push("/");
  await router.isReady();
});

describe("home", () => {
  it("shows selection even if the server already selected a workspace; creates, renames and enters", async () => {
    const pinia = (await import("pinia")).getActivePinia()!;
    const wrapper = mount(HomePage, { global: { plugins: [pinia, router] } });
    await settle();
    expect(wrapper.text()).toContain("Default");
    expect(wrapper.text()).toContain("Selected");
    expect(useNamedWorkspacesStore().activeId).toBeNull();
    await wrapper.get(".create-trigger").trigger("click");
    await wrapper.get("#new-name").setValue("Project A");
    await wrapper.get(".create-form").trigger("submit");
    await settle();
    expect(wrapper.text()).toContain("Project A");
    const card = wrapper.findAll(".card").find((c) => c.text().includes("Project A"))!;
    await card.get("button.title").trigger("click");
    await card.get("input[id^=name]").setValue("Renamed");
    await card.get("input[id^=name]").trigger("keydown.enter");
    await settle();
    expect(card.text()).toContain("0 loaded experiments");
    expect(card.text()).toContain("/logs");
    expect(card.find("form").exists()).toBe(false);
    await card.trigger("click");
    await settle();
    expect(useNamedWorkspacesStore().active?.name).toBe("Renamed");
    expect(router.currentRoute.value.name).toBe("studio");
    wrapper.unmount();
  });
});

describe("workspace card experiment previews", () => {
  it("shows each workspace's saved experiments with shortened names and their colours", async () => {
    const named = useNamedWorkspacesStore();
    await named.refresh();
    const other = await named.create("Other");
    const first = { ...emptyWorkspace(), experiments: ["logs/lle6-vdn", "logs/lle6-vdn-icm"], colours: { "logs/lle6-vdn": "#123456" } };
    localStorage.setItem(`${STORAGE_KEY}.default`, JSON.stringify(first));
    localStorage.setItem(
      `${STORAGE_KEY}.${encodeURIComponent(other.id)}`,
      JSON.stringify({ ...emptyWorkspace(), experiments: ["logs/other"] }),
    );

    const wrapper = mount(HomePage, { global: { plugins: [(await import("pinia")).getActivePinia()!, router] } });
    await settle();
    const cards = wrapper.findAll(".workspace-card");
    expect(cards[0].text()).toContain("2 loaded experiments");
    expect(cards[0].findAll(".preview-pill").map((pill) => pill.text())).toEqual(["vdn", "vdn-icm"]);
    expect(cards[0].get('.preview-pill[title="logs/lle6-vdn"] .dot').attributes("style")).toContain("#123456");
    expect(cards[1].text()).toContain("1 loaded experiment");
    expect(cards[1].findAll(".preview-pill").map((pill) => pill.text())).toEqual(["other"]);
    wrapper.unmount();
  });

  it("updates the active card from the plotting state before it is persisted", async () => {
    const named = useNamedWorkspacesStore();
    await named.refresh();
    await named.enter("default");
    const plotting = useWorkspaceStore();
    plotting.addExperiment("logs/new", "#abcdef");

    const wrapper = mount(HomePage, { global: { plugins: [(await import("pinia")).getActivePinia()!, router] } });
    await settle();
    expect(wrapper.get(".workspace-card").text()).toContain("1 loaded experiment");
    expect(wrapper.get(".preview-pill").attributes("title")).toBe("logs/new");
    plotting.removeExperiment("logs/new");
    await wrapper.vm.$nextTick();
    expect(wrapper.get(".workspace-card").text()).toContain("0 loaded experiments");
    wrapper.unmount();
  });
});
