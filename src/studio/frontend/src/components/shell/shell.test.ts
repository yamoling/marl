import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { defineComponent, h, nextTick, ref } from "vue";
import { escStack, EscStack, useEscClose } from "../../composables/useEscStack";
import { useToasts } from "../../stores/toasts";
import Segmented from "./Segmented.vue";

describe("Esc stack", () => {
  it("closes last opened first, dialogs before the rest", () => {
    const s = new EscStack();
    const log: string[] = [];
    s.push(() => log.push("drawer"));
    s.push(() => log.push("dialog"), 10);
    s.push(() => log.push("popover"));
    while (s.handle());
    expect(log).toEqual(["dialog", "popover", "drawer"]);
    expect(s.handle()).toBe(false);
  });

  it("useEscClose registers while open and Escape closes it", async () => {
    const open = ref(true);
    const C = defineComponent({
      setup() {
        useEscClose(open, () => (open.value = false));
        return () => h("div");
      },
    });
    const w = mount(C);
    expect(escStack.size).toBe(1);
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    expect(open.value).toBe(false);
    await nextTick();
    expect(escStack.size).toBe(0);
    open.value = true;
    await nextTick();
    w.unmount();
    expect(escStack.size).toBe(0);
  });
});

describe("toasts store", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    vi.useFakeTimers();
  });
  afterEach(() => vi.useRealTimers());

  it("auto-dismisses, keeps errors, runs actions", async () => {
    const t = useToasts();
    t.push("hello");
    const err = t.push({ message: "failed", level: "error" });
    const undo = vi.fn();
    const withAction = t.push({ message: "Deleted", actions: [{ label: "Undo", run: undo }] });
    vi.advanceTimersByTime(4000);
    expect(t.toasts.map((x) => x.message)).toEqual(["failed", "Deleted"]);
    await t.runAction(withAction, t.toasts[1].actions[0]);
    expect(undo).toHaveBeenCalled();
    expect(t.toasts.map((x) => x.id)).toEqual([err]);
  });
});

describe("Segmented", () => {
  it("emits the chosen value", async () => {
    const w = mount(Segmented, {
      props: {
        modelValue: "mean",
        options: [
          { value: "mean", label: "mean" },
          { value: "median", label: "median" },
        ],
      },
    });
    await w.findAll("button")[1].trigger("click");
    expect(w.emitted("update:modelValue")).toEqual([["median"]]);
    expect(w.find("button.on").text()).toBe("mean");
  });
});
