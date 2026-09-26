import { describe, expect, it } from "vitest";
import { assignColour, assignColours, MISSING_COLOUR, PALETTE, paramScale, rampColour, tableColour, tableDash } from "./colour";

describe("experiment colours", () => {
  it("assigns in load order, first free colour", () => {
    expect(assignColours(["a", "b", "c"], {})).toEqual({ a: PALETTE[0], b: PALETTE[1], c: PALETTE[2] });
  });

  it("is stable across unload/reload", () => {
    const all = assignColours(["a", "b", "c"], {});
    const remembered = { ...all };
    // unload b: others keep their colours
    const { b: _b, ...afterUnload } = all;
    const without = assignColours(["a", "c"], afterUnload, remembered);
    expect(without).toEqual({ a: PALETTE[0], c: PALETTE[2] });
    // reload b: it gets its previous colour back
    const reloaded = assignColours(["a", "c", "b"], without, remembered);
    expect(reloaded.b).toBe(PALETTE[1]);
    expect(reloaded.a).toBe(PALETTE[0]);
    expect(reloaded.c).toBe(PALETTE[2]);
  });

  it("a remembered colour taken by another experiment falls back to the first free one", () => {
    const current = { a: PALETTE[0], d: PALETTE[1] };
    expect(assignColour("b", current, { b: PALETTE[1] })).toBe(PALETTE[2]);
  });

  it("cycles the palette when all 10 colours are used", () => {
    const ids = Array.from({ length: 12 }, (_, i) => `e${i}`);
    const cs = assignColours(ids, {});
    expect(new Set(Object.values(cs).slice(0, 10)).size).toBe(10);
    expect(PALETTE).toContain(cs.e11);
  });
});

describe("tables, ramps and param scales", () => {
  it("fixed table colours and dashes; other tables are stable", () => {
    expect([tableColour("test"), tableColour("train"), tableColour("training_data")]).toEqual([PALETTE[0], PALETTE[1], PALETTE[4]]);
    expect(tableColour("test-policy-on-test-envs")).toBe(tableColour("test-policy-on-test-envs"));
    expect([tableDash("test"), tableDash("train"), tableDash("training_data"), tableDash("x")]).toEqual(["", "6 4", "2 3", "10 3 2 3"]);
  });

  it("ramp goes from light violet to deep indigo", () => {
    expect(rampColour(0)).toBe("rgb(178,160,255)");
    expect(rampColour(1)).toBe("rgb(52,24,150)");
  });

  it("param scale: numeric sorted, categorical first-seen, missing grey", () => {
    const num = paramScale("lr", [3e-4, 1e-4, undefined, 3e-4]);
    expect(num.numeric).toBe(true);
    expect(num.entries.map((e) => e.value)).toEqual([1e-4, 3e-4]);
    expect(num.missing).toBe(true);
    expect(num.colourOf(undefined)).toBe(MISSING_COLOUR);
    const cat = paramScale("mixer", ["QMix", "VDN", true]);
    expect(cat.numeric).toBe(false);
    expect(cat.entries.map((e) => e.colour)).toEqual([PALETTE[0], PALETTE[1], PALETTE[2]]);
  });
});
