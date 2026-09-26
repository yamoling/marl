<script setup lang="ts">
/**
 * Modal sheet with a scrim: `bottom` (episodes & replay) or `large` (centred, library).
 * Slots: `header` (replaces eyebrow/title), default (scrolling body), `footer`.
 */
import { ESC_PRIORITY, useEscClose } from "../../composables/useEscStack";
import Icon from "./Icon.vue";

const props = withDefaults(defineProps<{ open: boolean; variant?: "bottom" | "large"; title?: string; eyebrow?: string }>(), {
    variant: "bottom",
    title: "",
    eyebrow: "",
});
const emit = defineEmits<{ "update:open": [value: boolean] }>();
const close = () => emit("update:open", false);
useEscClose(() => props.open, close, ESC_PRIORITY.sheet);
</script>

<template>
    <Teleport to="body">
        <Transition name="fade">
            <div v-if="open" class="scrim" :class="variant" @click="close" />
        </Transition>
        <Transition :name="variant === 'bottom' ? 'sheet-up' : 'sheet-center'">
            <section v-if="open" class="sheet" :class="variant" role="dialog" aria-modal="true" :aria-label="title || eyebrow">
                <header class="sh-head">
                    <div class="grow">
                        <slot name="header">
                            <div v-if="eyebrow" class="eyebrow">{{ eyebrow }}</div>
                            <h2 v-if="title">{{ title }}</h2>
                        </slot>
                    </div>
                    <button class="close" aria-label="Close" @click="close"><Icon name="x" /></button>
                </header>
                <div class="sh-body"><slot /></div>
                <footer v-if="$slots.footer" class="sh-foot"><slot name="footer" /></footer>
            </section>
        </Transition>
    </Teleport>
</template>

<style scoped>
.scrim {
    position: fixed;
    inset: 0;
    background: var(--scrim);
    z-index: var(--z-scrim);
}
.sheet {
    position: fixed;
    z-index: var(--z-sheet);
    background: var(--card);
    box-shadow: var(--sh-sheet);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    border-radius: var(--r-sheet);
}
.scrim.bottom {
    z-index: calc(var(--z-sheet) + 1);
}
.sheet.bottom {
    z-index: calc(var(--z-sheet) + 2);
    left: 10px;
    right: 10px;
    bottom: 10px;
    height: min(580px, 74vh);
}
.sheet.large {
    top: 5vh;
    left: 50%;
    width: min(1120px, 94vw);
    height: 88vh;
    transform: translateX(-50%);
}
.sh-head {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 20px 22px 12px;
}
.sh-head h2 {
    margin: 2px 0 4px;
    font-size: 20px;
    letter-spacing: -0.02em;
}
.grow {
    flex: 1;
    min-width: 0;
}
.sh-body {
    flex: 1;
    overflow: auto;
    padding: 4px 22px 22px;
}
.sh-foot {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 22px;
    border-top: 1px solid var(--line);
    background: var(--foot);
}
.fade-enter-active,
.fade-leave-active {
    transition: opacity 0.2s;
}
.fade-enter-from,
.fade-leave-to {
    opacity: 0;
}
.sheet-up-enter-active,
.sheet-up-leave-active,
.sheet-center-enter-active,
.sheet-center-leave-active {
    transition:
        transform 0.28s cubic-bezier(0.2, 0.8, 0.2, 1),
        opacity 0.2s;
}
.sheet-up-enter-from,
.sheet-up-leave-to {
    transform: translateY(110%);
}
.sheet-center-enter-from,
.sheet-center-leave-to {
    transform: translate(-50%, 16px);
    opacity: 0;
}
</style>
