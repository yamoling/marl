/**
 * Lightweight tooltips: `v-tip="text"` stores the text in `data-tip`; one `TooltipLayer`
 * shows it on hover/focus (as in the Composer mockup). Disabled controls should be wrapped in
 * an element carrying the tooltip, since disabled buttons do not receive mouse events.
 */
import type { Directive } from "vue";

function apply(el: HTMLElement, text: unknown): void {
  const s = text === null || text === undefined || text === false ? "" : String(text);
  if (s) {
    el.dataset.tip = s;
    if (!el.getAttribute("aria-label") && !el.textContent?.trim()) el.setAttribute("aria-label", s);
  } else delete el.dataset.tip;
}

export const vTip: Directive<HTMLElement, unknown> = {
  mounted: (el, b) => apply(el, b.value),
  updated: (el, b) => apply(el, b.value),
};
