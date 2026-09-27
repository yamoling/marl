<script setup lang="ts" generic="T extends string | number">
/**
 * Segmented control (Composer `.seg`). Each option may carry a tooltip and a disabled reason;
 * the `param` variant uses the parameter colour for the selected segment.
 */
import type { SegmentOption } from "./types";

defineProps<{ options: SegmentOption<T>[]; modelValue: T; label?: string; dim?: boolean }>();
const emit = defineEmits<{ "update:modelValue": [value: T] }>();
</script>

<template>
    <div class="seg" :class="{ dim }" role="radiogroup" :aria-label="label">
        <button
            v-for="o in options"
            :key="String(o.value)"
            type="button"
            role="radio"
            :aria-checked="o.value === modelValue"
            :class="{ on: o.value === modelValue, pseg: o.variant === 'param' }"
            :title="o.title"
            :disabled="o.disabled"
            @click="o.value !== modelValue && emit('update:modelValue', o.value)"
        >
            <slot name="option" :option="o">{{ o.label }}</slot>
        </button>
    </div>
</template>

<style scoped>
.seg {
    display: inline-flex;
    background: var(--seg-bg);
    border-radius: 8px;
    padding: 2px;
    gap: 1px;
}
.seg.dim {
    opacity: 0.4;
}
.seg button {
    border: 0;
    background: transparent;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 11.5px;
    color: var(--ink2);
    transition: 0.12s;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    white-space: nowrap;
}
.seg button:hover:not(:disabled) {
    color: var(--ink);
}
.seg button:disabled {
    opacity: 0.45;
}
.seg button.on {
    background: var(--card);
    color: var(--acc2);
    box-shadow: 0 1px 3px rgba(20, 20, 50, 0.13);
    font-weight: 600;
}
.seg button.pseg.on {
    color: var(--param-ink);
}
</style>
